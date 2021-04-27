#include "mlir/Support/MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "Support/InitAllDialects.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;

    mlir::registerAllDialects(registry);

    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "PCC modular optimizer driver", registry,
                          /*prelaodDialectsInContext=*/false));
}