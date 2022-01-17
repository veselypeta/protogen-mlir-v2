//
// Created by petr on 10/01/2022.
//
#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include <mlir/IR/BlockAndValueMapping.h>

using namespace mlir;
using namespace mlir::fsm;

// Private Namespace for Implementation
namespace {

bool isTransientState(StateOp stateOp) {
  return stateOp.isTransient().hasValue() && stateOp.isTransient().getValue();
}
__attribute__((unused))
bool isTransientState(TransitionOp transitionOp) {
  return isTransientState(transitionOp->getParentOfType<StateOp>());
}
__attribute__((unused))
bool isCpuEvent(llvm::StringRef val) {
  return val == "load" || val == "store" || val == "evict";
}
__attribute__((unused))
llvm::StringRef getLastMessageSend(TransitionOp ) {
  // TODO - implement this
  return "";
}



StateOp getStableStartState(StateOp stateOp) {
  MachineOp parentMach = stateOp->getParentOfType<MachineOp>();
  // if a stable state -> return self
  if (!isTransientState(stateOp))
    return stateOp;

  assert(stateOp.prevTransition().hasValue() && "no linked previous state!");
  TransitionOp prevTransition =
      parentMach.lookupSymbol<TransitionOp>(stateOp.prevTransitionAttr());
  assert(prevTransition != nullptr &&
         "could not lookup the correct link from prevTransition");
  return getStableStartState(prevTransition->getParentOfType<StateOp>());
}
__attribute__((unused))
StateOp getStableStartState(TransitionOp transitionOp) {
  return getStableStartState(transitionOp->getParentOfType<StateOp>());
}

// We need a custom rewriter to be able to construct it
class ProtoGenRewriter : public PatternRewriter {
public:
  explicit ProtoGenRewriter(MLIRContext *ctx) : PatternRewriter{ctx} {
    // TODO - overwrite necessary methods here
  }
};

// Create the Optimization Pass
class StallingOptimizationPass
    : public StallingOptimizationPassBase<StallingOptimizationPass> {
public:
  void runOnOperation() override;
};

// Implement the runOnOperation function
void StallingOptimizationPass::runOnOperation() {
  /// grab the module/cache/directory
  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();
  __attribute__((unused))
  MachineOp theCache = theModule.lookupSymbol<MachineOp>("cache");
  __attribute__((unused))
  MachineOp theDirectory = theModule.lookupSymbol<MachineOp>("directory");

  ProtoGenRewriter rewriter(&getContext());

  // walk each transient state
  auto result = theModule.walk([&](StateOp startState) {
    // Skip non-transient states
    if (!isTransientState(startState))
      return WalkResult::advance();

    // Now we need to find the stable start state
    StateOp logicalStartState = getStableStartState(startState);

    // find out which message we have sent to the directory
    auto racingResult =
        logicalStartState.walk([&](TransitionOp racingTransaction) {
          // skip if we can already handle this message in the current state
          if (!startState.lookupSymbol<TransitionOp>(
                  racingTransaction.sym_name()))
            return WalkResult::advance();

          // set the rewriter to create a new transition at the end of the
          // current state
          rewriter.setInsertionPointToEnd(startState.getBody());

          // this new transition will be
          // (startState, racingTransition.action, ???)
          // currently, we do not specify a nextState
          // This we will deduce later
          TransitionOp optTrans = rewriter.create<TransitionOp>(
              startState.getLoc(), racingTransaction.sym_nameAttr().getValue(),
              racingTransaction.getType(),
              /*nextState*/ nullptr);
          Block *optEntry = optTrans.addEntryBlock();
          rewriter.setInsertionPointToStart(optEntry);
          // create a dummy op which is later replaced
          auto inlinerOp = rewriter.create<ConstantOp>(
              optTrans.getLoc(), rewriter.getI64IntegerAttr(1));

          // we have to inline the contents of the racing transaction into this
          // new transaction i.e. copy the actions
          InlinerInterface inliner(&getContext());
          BlockAndValueMapping mapping;

          LogicalResult wasInlined = inlineRegion(
              /*inliner interface*/ inliner,
              /* src region */ &racingTransaction.body(),
              /* inline point */ inlinerOp,
              /* mapper */ mapping,
              /* results to replace */ {},
              /* region result types */ {});

          assert(succeeded(wasInlined) && "failed to inline Transition");

          // remove the unused inline const op
          rewriter.eraseOp(inlinerOp);



          ///// Now we need to find out which state to transition to /////
//          if(racingTransaction.nextState().hasValue())
//            ;




          // advance to next transition
          return WalkResult::advance();
        });

    if (racingResult.wasInterrupted())
      signalPassFailure();
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();
}

} // namespace

namespace mlir {
namespace fsm {

std::unique_ptr<OperationPass<ModuleOp>> createStallingOptimizationPass() {
  return std::make_unique<StallingOptimizationPass>();
}

} // namespace fsm
} // namespace mlir