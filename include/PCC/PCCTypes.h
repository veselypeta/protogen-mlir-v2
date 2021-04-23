#pragma once
#include "mlir/IR/Types.h"
#include "PCC/PCCDialect.h"
#include "PCC/PCCOps.h"

namespace mlir
{

    namespace pcc
    {
        // Types
        class IDType;

        // this is a common base for all PCC Types
        class PCCType : public Type
        {
            void print(raw_ostream &os) const;

            // Support method to enable LLVM-style type casting.
            static bool classof(Type type)
            {
                return llvm::isa<mlir::pcc::PCCDialect>(type.getDialect());
            }

        protected:
            using Type::Type;
        };

        ///
        class IDType : public Type::TypeBase<IDType, Type, TypeStorage>
        {
            using Base::Base;
            static IDType get(MLIRContext *context) { return Base::get(context); }
        };

    } // namespace pcc
} // namespace mlir
