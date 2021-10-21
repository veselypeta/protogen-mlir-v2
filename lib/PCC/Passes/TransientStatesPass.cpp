#include "PCC/Passes/Passes.h"
#include "PassDetail.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include <algorithm>
#include <mlir/IR/BlockAndValueMapping.h>

using namespace mlir;
using namespace llvm;
using namespace mlir::pcc;

class ProcessOpPattern : public OpRewritePattern<ProcessOp> {
public:
  ProcessOpPattern(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<ProcessOp>(ctx, benefit) {}

  void initialize() {}
};

class TransientStatesRewriter : public PatternRewriter {
public:
  explicit TransientStatesRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
  // TODO - override any necessary methods here
};

namespace {

struct TransientStatesPass
    : public pcc::TransientStatesBase<TransientStatesPass> {
  void runOnOperation() override;
};

void TransientStatesPass::runOnOperation() {
  ModuleOp module = mlir::OperationPass<ModuleOp>::getOperation();

  WalkResult result = module.walk([&](ProcessOp procOp) {
    // This is effectively the matcher for process ops which contain a nested
    // Transaction Op
    auto rewrite_if_valid = [&](ProcessOp &op,
                                auto rewrite_cb) -> LogicalResult {
      TransientStatesRewriter rewriter(&getContext());
      auto is_transaction_op = [](Operation &op) {
        return mlir::isa<TransactionOp>(op);
      };
      auto nestedOps = op.getBody().getOps();
      auto result =
          std::find_if(nestedOps.begin(), nestedOps.end(), is_transaction_op);

      if (result != nestedOps.end()) {
        TransactionOp transOp = mlir::dyn_cast<TransactionOp>(*result);
        return rewrite_cb(transOp, rewriter);
      }
      return success();
    };

    // This is the rewriter code
    auto transient_states_rewriter =
        [](TransactionOp &op, PatternRewriter &rewriter) -> LogicalResult {
      ProcessOp parentProcess = op->getParentOfType<ProcessOp>();
      // insert a new set state operation before
      rewriter.setInsertionPointAfter(parentProcess);
      // TODO - figure out the action

      // Workaround just append the name of the action to the current state name
      StringRef procName = parentProcess.sym_name();
      auto nestedOpsIt = op.getRegion().getOps<TransitionOp>();
      std::for_each(
          nestedOpsIt.begin(), nestedOpsIt.end(),
          [&](TransitionOp transitionOp) {
            // Create the new Op
            std::string newProcName =
                (procName + "_" + transitionOp.guard()).str();
            Type machType = parentProcess.getType().getInput(0);
            Type msgType = transitionOp.msgType();
            FunctionType newFuncType =
                rewriter.getFunctionType({machType, msgType}, llvm::None);
            ProcessOp newProc = rewriter.create<ProcessOp>(
                transitionOp.getLoc(), newProcName, newFuncType);
            // create a constant op to facilitate the inlining
            Block *entry = newProc.addEntryBlock();
            rewriter.setInsertionPointToStart(entry);
            auto inlineOp = rewriter.create<InlineConstOp>(
                newProc.getLoc(), rewriter.getI64IntegerAttr(4));

            // inline the regions
            InlinerInterface inliner(rewriter.getContext());

            // create a mapper for the arguments to the region
            BlockAndValueMapping mapper;
            mapper.map(parentProcess.getArgument(0),
                       newProc.getArgument(0)); // map the mach ref
            mapper.map(transitionOp.body().getArgument(0),
                       newProc.getArgument(0)); // map the msg ref

            LogicalResult wasInlined = inlineRegion(
                /* inliner interface */ inliner,
                /* src region */ &transitionOp.getRegion(),
                /* inline point */ inlineOp,
                /* mapper */ mapper,
                /* results to replace */ {},
                /* region result types */ {});

            assert(succeeded(wasInlined) && "Failed to inline PCC function");

            // remove the unnecessary inline const op
            rewriter.eraseOp(inlineOp);

          });

      // TODO - Insert a set-state operation
      //StringRef nextState = parentProcess.sym_name();

      rewriter.eraseOp(op);
      return success();
    };

    if (failed(rewrite_if_valid(procOp, transient_states_rewriter)))
      procOp.emitError("Failed to successfully rewrite a process Op");

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> pcc::createTransientStatesPass() {
  return std::make_unique<TransientStatesPass>();
}