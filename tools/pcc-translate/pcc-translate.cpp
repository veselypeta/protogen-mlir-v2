#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "RegisterPCCTranslations.h"

int main(int argc, char **argv) {
  // from PCC translation
  mlir::registerPCCToPCCDialectTranslation();

  // from FSM translation
  mlir::registerPCCToFSMDialectTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "PCC Translation tool"));
}