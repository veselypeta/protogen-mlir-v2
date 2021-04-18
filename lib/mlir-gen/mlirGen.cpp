#include "mlir-gen/mlirGen.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"


namespace
{
    class MLIRGenImpl
    {

    public:
        MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
        mlir::ModuleOp mlirGen(ProtoCCParser::DocumentContext *ctx){
            return theModule;
        }
    private:
    mlir::ModuleOp theModule;
    mlir::OpBuilder builder;
    };
} // namespace


namespace pcc
{
    mlir::ModuleOp mlirGen(mlir::MLIRContext &mlirCtx, ProtoCCParser::DocumentContext *docCtx) {
        return MLIRGenImpl(mlirCtx).mlirGen(docCtx);
    }
} // namespace pcc
