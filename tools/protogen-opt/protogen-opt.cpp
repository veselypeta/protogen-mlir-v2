#include "PCC/Passes/Passes.h"
#include "FSM/Passes/Passes.h"
#include "Support/InitAllDialects.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  // register all necessary dialects
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  // initialise PCC passes
  mlir::pcc::initAllPCCPasses();
  mlir::pcc::initAllPipelines();

  // initialise FSM passes
  mlir::fsm::registerAllFSMPasses();
  mlir::fsm::registerAllFSMPipelines();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "PCC modular optimizer driver", registry,
                        /*preloadDialectsInContext=*/false));
}