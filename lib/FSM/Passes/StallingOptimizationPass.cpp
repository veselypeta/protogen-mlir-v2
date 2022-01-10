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
class ProtoGenRewriter : public PatternRewriter{
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
  // grab the module
  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();

  // then walk each operation

}




}

namespace mlir {
namespace fsm {

std::unique_ptr<OperationPass<ModuleOp>> createStallingOptimizationPass() {
  return nullptr;
}

} // namespace fsm
} // namespace mlir