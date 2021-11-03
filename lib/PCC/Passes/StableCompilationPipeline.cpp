#include "PCC/Passes/Passes.h"
#include "PassDetail.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::pcc;

namespace {

void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(mlir::pcc::createEndStatesPass());
  pm.addPass(mlir::pcc::createMutexOpsPass());
  pm.addPass(mlir::pcc::createTransientStatesPass());
}

} // namespace

namespace mlir {
namespace pcc {

void registerStableCompilation() {
  PassPipelineRegistration<>("compile-stable",
                             "compile the raw output from pcc-translate into a "
                             "verifiable stable state protocol",
                             pipelineBuilder);
}

} // namespace pcc
} // namespace mlir