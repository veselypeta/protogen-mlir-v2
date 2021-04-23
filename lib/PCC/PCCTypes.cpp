#include "PCC/PCCTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir
{
    namespace pcc
    {

        using mlir::TypeStorageAllocator;

        void PCCType::print(raw_ostream &os) const
        {
            TypeSwitch<PCCType>(*this)
                .Case<IDType>([&](Type) {os << "ID";});
        }

        Type PCCDialect::parseType(::mlir::DialectAsmParser &parser) const {
            // TODO
        }

        void PCCDialect::printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const {
            // TODO
        }

    } // namespace pcc
} // namespace mlir
