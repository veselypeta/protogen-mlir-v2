#ifndef PCC_TYPES
#define PCC_TYPES


def PCCType : DialectType<PCC_Dialect, CPred<"$_self.isa<PCCType>()">, 
    "PCCType", "::mlir::pcc::PCCType">;

def IDType : DialectType<PCC_Dialect, CPred<"$_self.isa<IDType>()">,
    "id", "::mlir::pcc::IDType">, BuildableType<"IDType::get($_builder.getContext())">; 


def NetworkType : DialectType<PCC_Dialect, CPred<"$_self.isa<NetworkType>()">,
    "network", "::mlir::pcc::NetworkType">, BuildableType<"NetworkType::get($_builder.getContext())">;

#endif //PCC_TYPES