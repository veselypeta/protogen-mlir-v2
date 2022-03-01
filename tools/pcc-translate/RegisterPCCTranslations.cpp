#include "FSM/FSMDialect.h"
#include "PCC/PCCDialect.h"
#include "RegisterPCCTranslations.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Translation.h"

#include "ProtoCCLexer.h"
#include "ProtoCCParser.h"
#include "mlir-gen/mlirGen.h"
#include <iostream>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
namespace {

template <class MLIRGenerator>
mlir::ModuleOp generateMLIR(const llvm::MemoryBuffer *inputBuf,
                            mlir::MLIRContext *ctx, MLIRGenerator generator) {
  // get the input filename
  llvm::StringRef filename = inputBuf->getBufferIdentifier();
  std::string filenameStr(filename.data());

  // read the file as a string
  llvm::StringRef strRef = inputBuf->getBuffer();
  std::string inputFileStr(strRef.data());

  // generate lexer and parser objects from the string
  antlr4::ANTLRInputStream input(inputFileStr);
  ProtoCCLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  ProtoCCParser parser(&tokens);

  // get the document ctx and return mlirGen
  ProtoCCParser::DocumentContext *tree = parser.document();
  return generator(*ctx, tree, filenameStr);
}

} // namespace

namespace mlir {
void registerPCCToPCCDialectTranslation() {

  // Source Manager Version
  mlir::TranslateToMLIRRegistration registration(
      "pcc-to-pcc-dialect",
      [](llvm::SourceMgr &manager, mlir::MLIRContext *ctx) {
        ctx->getOrLoadDialect<mlir::StandardOpsDialect>();
        ctx->getOrLoadDialect<mlir::pcc::PCCDialect>();
        return generateMLIR(manager.getMemoryBuffer(manager.getMainFileID()),
                            ctx, mlir::pcc::mlirGen);
      });
}

void registerPCCToFSMDialectTranslation() {
  mlir::TranslateToMLIRRegistration(
      "pcc-to-fsm-dialect",
      [&](llvm::SourceMgr &manager, mlir::MLIRContext *ctx) -> ModuleOp {
        ctx->getOrLoadDialect<mlir::StandardOpsDialect>();
        ctx->getOrLoadDialect<mlir::fsm::FSMDialect>();
        return generateMLIR(manager.getMemoryBuffer(manager.getMainFileID()),
                            ctx, mlir::fsm::mlirGen);
      });
}
} // namespace mlir
