#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/MlirOptMain.h"
#include "FSM/Passes/Passes.h"

#include "FSM/FSMDialect.h"
int main(int argc, char **argv) {
  using namespace mlir;
  using namespace mlir::fsm;

  DialectRegistry registry;
  registry.insert<FSMDialect>();
  registry.insert<StandardOpsDialect>();

  // init all passes/pipelines
  registerAllFSMPasses();

  return failed(mlir::MlirOptMain(argc, argv, "FSM modular optimizer driver",
                                  registry,
                                  /*preloadDialectsInContext=*/false));
}