#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::pcc;

namespace {

bool is_cpu_evt_proc(ProcessOp &processOp) {
  llvm::StringRef actionAttrVal =
      processOp->getAttr("action").cast<StringAttr>().getValue();
  return actionAttrVal == "load" || actionAttrVal == "store" ||
         actionAttrVal == "evict";
}

class ReleaseMutexOnBreakOp : public OpRewritePattern<BreakOp> {
public:
  ReleaseMutexOnBreakOp(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<BreakOp>(ctx, benefit) {}

  LogicalResult matchAndRewrite(BreakOp op,
                                PatternRewriter &rewriter) const override {
    // skip if we have already inserted a release
    auto opsIt = op->getParentOfType<TransitionOp>().body().getOps();
    if (std::find_if(std::begin(opsIt), std::end(opsIt), [](Operation &op) {
          // if cast succeeds then there already exists a release op
          if (dyn_cast<ReleaseOp>(op) != nullptr)
            return true;
          return false;
        }) != opsIt.end())
      return failure();

    rewriter.setInsertionPoint(op);
    rewriter.create<ReleaseOp>(op.getLoc());
    return success();
  }
};

class MutexRewriter : public PatternRewriter {
public:
  explicit MutexRewriter(MLIRContext *ctx) : PatternRewriter{ctx} {}
  // TODO - override necessary methods here
};

class MutexOpsPass : public mlir::pcc::MutexOpsPassBase<MutexOpsPass> {
  void runOnOperation() override;
};

void MutexOpsPass::runOnOperation() {
  ModuleOp moduleOp = OperationPass<ModuleOp>::getOperation();

  WalkResult result = moduleOp.walk([&](ProcessOp processOp) {
    // skip directory (only insert mutexes in cache operations) and CPU event
    llvm::StringRef startState =
        processOp->getAttr("start_state").cast<StringAttr>().getValue();
    if (!is_cpu_evt_proc(processOp) ||
        startState.find("directory") != llvm::StringRef::npos)
      return WalkResult::advance();

    MutexRewriter rewriter{&getContext()};

    // point to start of entry block
    Block *entryBlock = &processOp.getBody().front();
    rewriter.setInsertionPointToStart(entryBlock);

    // for each transaction (process) insert an 'acquire' at the beginning
    rewriter.create<AcquireOp>(processOp.getLoc());

    // find the end of the transaction (there may be multiple paths) and insert
    // a release op

    // RULES:
    // (1) if we have a transaction op then we insert a release before each
    // break operation (2) if not transaction op exists then we release at the
    // end of the block

    auto transactionOps = entryBlock->getOps<TransactionOp>();
    if (transactionOps.empty()) { // Rule (2)
      rewriter.setInsertionPointToEnd(entryBlock);
      rewriter.create<ReleaseOp>(processOp.getLoc());
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    signalPassFailure();

  // apply the patterns for break op to implement Rule (1) from above
  OwningRewritePatternList patterns(&getContext());
  patterns.add<ReleaseMutexOnBreakOp>(&getContext(), 1);

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns))))
    signalPassFailure();
}

} // namespace

namespace mlir {
namespace pcc {
std::unique_ptr<OperationPass<ModuleOp>> createMutexOpsPass() {
  return std::make_unique<MutexOpsPass>();
}
} // namespace pcc
} // namespace mlir