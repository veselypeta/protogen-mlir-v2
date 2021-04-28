#include "TranslatePCCToMLIR.h"
#include "PCC/PCCOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Translation.h"

#include "ProtoCCLexer.h"
#include "ProtoCCParser.h"
#include "mlir-gen/mlirGen.h"
#include <iostream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

mlir::ModuleOp PCCMLIRGen(const llvm::MemoryBuffer *inputBuf,
                          mlir::MLIRContext *ctx) {
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
  return pcc::mlirGen(*ctx, tree, filenameStr);
}

namespace mlir {
void registerPCCToMLIRTranslation() {
  // mlir::TranslateToMLIRRegistration registration("pcc-to-mlir",
  // [](llvm::StringRef strRef, mlir::MLIRContext *ctx) {
  //     ctx->getOrLoadDialect<mlir::StandardOpsDialect>();
  //     ctx->getOrLoadDialect<mlir::pcc::PCCDialect>();
  //     return PCCMLIRGen(strRef, ctx);
  // });

  // Source Manager Version
  mlir::TranslateToMLIRRegistration registration(
      "pcc-to-mlir", [](llvm::SourceMgr &manager, mlir::MLIRContext *ctx) {
        ctx->getOrLoadDialect<mlir::StandardOpsDialect>();
        ctx->getOrLoadDialect<mlir::pcc::PCCDialect>();
        return PCCMLIRGen(manager.getMemoryBuffer(manager.getMainFileID()),
                          ctx);
      });
}
} // namespace mlir
