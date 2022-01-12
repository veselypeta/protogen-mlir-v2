//
// Created by petr on 10/01/2022.
//
#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::fsm;


// Private Namespace for Implementation
namespace {

// We need a custom rewriter to be able to construct it
class ProtoGenRewriter : public PatternRewriter {
  explicit ProtoGenRewriter(MLIRContext *ctx) : PatternRewriter{ctx}{

    // TODO - overwrite necessary methods here

  }
};

// Create the Optimization Pass
class StallingOptimizationPass : public StallingOptimizationPassBase<StallingOptimizationPass>{
public:
  void runOnOperation() override;
};

// Implement the runOnOperation function
void StallingOptimizationPass::runOnOperation() {
  // grab the module/cache/directory
  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();
  MachineOp theCache = theModule.lookupSymbol<MachineOp>("cache");
  MachineOp theDirectory = theModule.lookupSymbol<MachineOp>("directory");

  /// Useful Lambdas ///
  auto is_transient_state = [](TransitionOp &op){
    // TODO - Implement how to detect a transient state
    return true;
  };

  auto get_stable_start_state = [](TransitionOp &op){
    // TODO - Implement Properly
    return op.sym_name();
  };

  auto get_possible_racing_transactions = [&](llvm::StringRef startState){
    auto stableStateRef = theCache.lookupSymbol<StateOp>(startState);

  };


  auto result = theModule.walk([&](TransitionOp transitionOp){
    // Skip non-transient states
    if(!is_transient_state(transitionOp))
      return WalkResult::advance();

    // Now we need to find the stable start state
    llvm::StringRef stableStartState = get_stable_start_state(transitionOp);

    // find out which message we have sent to the directory

    // find the messages that can arrive in the stable start state







    return WalkResult::advance();
  });

  if(result.wasInterrupted())
    return signalPassFailure();
}




}

namespace mlir {
namespace fsm {

std::unique_ptr<OperationPass<ModuleOp>> createStallingOptimizationPass() {
  return std::make_unique<StallingOptimizationPass>();
}

} // namespace fsm
} // namespace mlir