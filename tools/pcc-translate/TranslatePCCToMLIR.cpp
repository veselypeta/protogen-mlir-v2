#include "TranslatePCCToMLIR.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Translation.h"
#include "PCC/PCCOps.h"

#include "mlir-gen/mlirGen.h"
#include <iostream>
#include "ProtoCCParser.h"
#include "ProtoCCLexer.h"

mlir::ModuleOp PCCMLIRGen(llvm::StringRef strRef, mlir::MLIRContext *ctx)
{
    // read the file as a string
    const unsigned char *s = strRef.bytes_begin();
    auto ss = reinterpret_cast<const char *>(s);
    std::string inputFileStr(ss);

    // generate lexer and parser objects from the string
    antlr4::ANTLRInputStream input(inputFileStr);
    ProtoCCLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    ProtoCCParser parser(&tokens);

    // get the document ctx and return mlirGen
    ProtoCCParser::DocumentContext *tree = parser.document();
    return pcc::mlirGen(*ctx, tree);
}

namespace mlir
{
    void registerPCCToMLIRTranslation()
    {
        mlir::TranslateToMLIRRegistration registration("pcc-to-mlir", [](llvm::StringRef strRef, mlir::MLIRContext *ctx) {
            ctx->getOrLoadDialect<mlir::StandardOpsDialect>();
            ctx->getOrLoadDialect<mlir::pcc::PCCDialect>();
            return PCCMLIRGen(strRef, ctx);
        });
    }
} // namespace mlir
