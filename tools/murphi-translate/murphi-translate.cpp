#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "translation/murphi/Translate.h"

int main(int argc, char **argv) {

  // translate pcc dialect to murphi
  mlir::registerPccToMurphiTranslation();

  // translate fsm dialect to murphi
  mlir::registerFsmToMurphiTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "Murphi Translation Tool"));
}