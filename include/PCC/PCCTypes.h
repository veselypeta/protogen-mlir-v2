#pragma once
#include "PCC/PCCDialect.h"
#include "mlir/IR/Types.h"

namespace pcc
{

    class PCCType : public Type
    {
        void print(raw_ostream &os) const;

        // Support method to enable LLVM-style type casting.
        static bool classof(Type type)
        {
            return llvm::isa<PCCDialect>(type.getDialect());
        }

    protected:
        using Type::Type;
    }

} // namespace pcc
