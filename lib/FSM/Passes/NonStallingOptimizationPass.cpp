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
                                PatternRewriter &rewriter) {
  //  auto theCache = transientState->getParentOfType<MachineOp>();
  // find out what racing msgs can arrive ???
  auto logicalEndStateOpt =
      utils::getNonStallingEndStateIfPossible(transientState);
  if (!logicalEndStateOpt.hasValue())
    return WalkResult::advance();
  auto logicalEndState = logicalEndStateOpt.getValue();
  emitRemark(transientState->getLoc(),
             "Non-Stalling: Found end state for transient state " +
                 transientState.sym_name() + " = " +
                 logicalEndState.sym_name());
  auto iRacingTransitions = logicalEndState.getOps<TransitionOp>();
  for (auto racingTransition : iRacingTransitions) {
    if (utils::isCpuEvent(racingTransition) ||
        transientState.lookupSymbol<TransitionOp>(
            racingTransition.sym_name()) != nullptr)
      continue;
    emitRemark(transientState.getLoc(),
               "Can optimize racing transition " + racingTransition.sym_name() +
                   " in state " + transientState.sym_name());

    // How do we optimize the non-stalling case ??

    // we need to create the new state to transition to
    // naming = {current_state}_{racing_action}_{prev_end}_{end_state}
    auto getNonStallingTransientStateName =
        [](llvm::StringRef startState, llvm::StringRef racing_action,
           llvm::StringRef prev_end, llvm::StringRef end_state) -> std::string {
      return (startState + "_" + racing_action + "_" + prev_end + "_" +
              end_state)
          .str();
    };
    auto nonStallStateName = getNonStallingTransientStateName(
        transientState.sym_name(), racingTransition.sym_name(),
        logicalEndState.sym_name(),
        racingTransition.nextStateAttr().getLeafReference());
    emitRemark(transientState.getLoc(), "Will transition from - " +
                                            transientState.sym_name() + " -> " +
                                            nonStallStateName);

    // create the transition
    rewriter.setInsertionPointToEnd(transientState.getBody());
    FunctionType nsFnTy = racingTransition.getType();
    auto nonStallTransition = rewriter.create<TransitionOp>(
        transientState.getLoc(), racingTransition.sym_name(), nsFnTy,
        rewriter.getSymbolRefAttr(nonStallStateName));
    // inline the racing transaction
    auto was_inlined = utils::inlineTransition(racingTransition, nonStallTransition, rewriter);
    if(failed(was_inlined)){
      emitError(nonStallTransition.getLoc(), "Failed to inline!");
      return WalkResult::interrupt();
    }
    // instead of send we need to defer


//    StateOp nonStallState =
//        rewriter.create<StateOp>(transientState.getLoc(), nonStallStateName);
  }

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