#include "PCC/Passes/Passes.h"
#include "PassDetail.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include <algorithm>

using namespace mlir;
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

      if (result != nestedOps.end()){
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
      rewriter.setInsertionPoint(op);
      // TODO - figure out the action
      StateType endState = StateType::get(rewriter.getContext(), "TransitionState");
      rewriter.create<InlineConstOp>(op.getLoc(), endState);
//      rewriter.create<StateUpdateOp>(op.getLoc(), );

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