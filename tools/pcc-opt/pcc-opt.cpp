#include "mlir/Support/MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "Support/InitAllDialects.h"
#include "PCC/PCCDialect.h"
#include "PCC/Passes/Passes.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;

    registry.insert<mlir::pcc::PCCDialect>();
    mlir::registerAllDialects(registry);

    mlir::pcc::initAllPCCPasses();

    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "PCC modular optimizer driver", registry,
                          /*preloadDialectsInContext=*/false));
}