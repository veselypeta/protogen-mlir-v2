#include "PCC/Passes/Passes.h"
#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::pcc;

namespace {

void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(mlir::pcc::createEndStatesPass());
  pm.addPass(mlir::pcc::createTransientStatesPass());
  pm.addPass(mlir::pcc::createStallingProtogenOptimizationPass());
}

} // namespace

namespace mlir {
namespace pcc {
void registerStallingCompilation() {
  PassPipelineRegistration<>("compile-stalling",
                             "compile the raw output from pcc-translate into "
                             "an optimised protocol which uses stalling",
                             pipelineBuilder);
}
} // namespace pcc
} // namespace mlir