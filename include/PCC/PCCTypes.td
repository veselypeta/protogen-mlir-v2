#ifndef PCC_TYPES
#define PCC_TYPES


def PCCType : DialectType<PCC_Dialect, CPred<"$_self.isa<PCCType>()">, 
    "PCCType", "::mlir::pcc::PCCType">;

def IDType : DialectType<PCC_Dialect, CPred<"$_self.isa<IDType>()">,
    "ID", "::mlir::pcc::IDType">, BuildableType<"IDType::get($_builder.getContext())">; 



#endif //PCC_TYPES