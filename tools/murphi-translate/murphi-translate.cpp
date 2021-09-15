#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "translation/murphi/Translate.h"

int main(int argc, char **argv){

  mlir::registerToMurphiTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "Murphi Translation Tool")
      );
}