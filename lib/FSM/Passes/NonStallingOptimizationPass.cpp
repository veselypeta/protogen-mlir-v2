#include "FSM/FSMUtils.h"
#include "FSM/Passes/Passes.h"
#include "PassDetail.h"
#include "ProtoGenRewriter.h"

using namespace mlir;
using namespace mlir::fsm;
namespace {

class NonStallingOptimizationPass
    : public mlir::fsm::NonStallingOptimizationPassBase<
          NonStallingOptimizationPass> {
public:
  void runOnOperation() override;
};

WalkResult handleTransientState(StateOp transientState,
                                PatternRewriter & /*rewriter*/) {
  emitRemark(transientState->getLoc(), transientState.sym_name().str());
  return WalkResult::advance();
}

void NonStallingOptimizationPass::runOnOperation() {
  // get links to ops
  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();
  MachineOp theCache = theModule.lookupSymbol<MachineOp>("cache");

  // create a rewriter
  ProtoGenRewriter rewriter(&getContext());

  auto result =
      utils::runOnEachTransientState(theCache, rewriter, handleTransientState);
  if (result.wasInterrupted())
    return signalPassFailure();
}

} // namespace

namespace mlir {
namespace fsm {
std::unique_ptr<OperationPass<ModuleOp>> createNonStallingOptimizationPass() {
  return std::make_unique<NonStallingOptimizationPass>();
}
} // namespace fsm
} // namespace mlir