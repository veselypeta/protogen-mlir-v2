#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "TranslatePCCToMLIR.h"

int main(int argc, char **argv)
{
    mlir::registerAllTranslations();

    mlir::registerPCCToMLIRTranslation();

    return failed(
        mlir::mlirTranslateMain(argc, argv, "PCC Translation tool"));
}