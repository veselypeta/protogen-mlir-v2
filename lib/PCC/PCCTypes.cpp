#include "PCC/PCCTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Builders.h"

/// Types
/// ::= id
/// Parse a type registered to this dialect.
using mlir::TypeStorageAllocator;
using namespace mlir;
using namespace mlir::pcc;




// Type Printer for PCC Types
// Uses the TypeSwitch Class to help printing types
void PCCType::print(raw_ostream &os) const
{
    // Not sure what this does
    // auto printWdithQualifier = [&](Optional<int32_t> width) {
    //     if (width)
    //         os << '<' << width.getValue() << '>';
    // };

    TypeSwitch<PCCType>(*this)
        .Case<IDType>([&](Type) { os << "id"; })
        .Default([](Type) { assert(0 && "unkown dialect type to print!"); });
}

static ParseResult parseType(PCCType &result, DialectAsmParser &parser)
{
    StringRef name;
    if (parser.parseKeyword(&name))
        return failure();

    MLIRContext *context = parser.getBuilder().getContext();

    if (name.equals("clock"))
    {
        // odd comma syntax
        return result = IDType::get(context), success();
    }
    return parser.emitError(parser.getNameLoc(), "unknown pcc type"),
           failure();
}

Type PCCDialect::parseType(::mlir::DialectAsmParser &parser) const
{
    PCCType result;
    if (::parseType(result, parser))
        return Type();

    return result;
}

void PCCDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &os) const
{
    type.print(os.getStream());
}
