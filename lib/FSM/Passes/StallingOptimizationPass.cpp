//
// Created by petr on 10/01/2022.
//
#include "FSM/FSMUtils.h"
#include "FSM/Passes/Passes.h"
#include "PassDetail.h"
#include "ProtoGenRewriter.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include <mlir/IR/BlockAndValueMapping.h>

using namespace mlir;
using namespace mlir::fsm;
using namespace mlir::fsm::utils;

// Private Namespace for Implementation
namespace {

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
  MachineOp theCache = theModule.lookupSymbol<MachineOp>("cache");
  ProtoGenRewriter rewriter(&getContext());

  // walk each transient state in the cache
  auto result = theCache.walk([&](StateOp startState) {
    // Skip non-transient states
    if (!isTransientState(startState))
      return WalkResult::advance();

    // Now we need to find the stable start state
    StateOp logicalStartState = getStableStartState(startState);

    // walk every action in the logical start state
    // since they can be a racing event
    auto racingResult =
        logicalStartState.walk([&](TransitionOp racingTransaction) {
          // skip if we can already handle this message in the current state
          // or is a cpu event
          llvm::StringRef racingEvent = racingTransaction.sym_name();
          if (startState.lookupSymbol<TransitionOp>(racingEvent) ||
              isCpuEvent(racingEvent)) {
            emitRemark(startState.getLoc(), "Skipping: (" +
                                                startState.sym_name() + ", " +
                                                racingEvent + ")");
            return WalkResult::advance();
          }

          emitRemark(startState.getLoc(), "Optimizing: (" +
                                              startState.sym_name() + ", " +
                                              racingEvent + ")");

          assert(succeeded(optimizeStateTransitionV2(
                     startState, racingTransaction, rewriter)) &&
                 "failed to optimize protocol");

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