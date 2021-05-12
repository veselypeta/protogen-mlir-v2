#ifndef PCC_TYPES
#define PCC_TYPES


def PCCType : DialectType<PCC_Dialect, CPred<"$_self.isa<PCCType>()">, 
    "PCCType", "::mlir::pcc::PCCType">;

def IDType : DialectType<PCC_Dialect, CPred<"$_self.isa<IDType>()">,
    "id", "::mlir::pcc::IDType">, BuildableType<"IDType::get($_builder.getContext())">;

def StateType : DialectType<PCC_Dialect, CPred<"$_self.isa<StateType>()">,
    "state", "::mlir::pcc::StateType">, BuildableType<"StateType::get($_builder.getContext())">; 


def NetworkType : DialectType<PCC_Dialect, CPred<"$_self.isa<NetworkType>()">,
    "network", "::mlir::pcc::NetworkType">, BuildableType<"NetworkType::get($_builder.getContext())">;


def SetType : DialectType<PCC_Dialect, CPred<"$_self.isa<SetType>()">,
    "set", "::mlir::pcc::SetType">, BuildableType<"SetType::get($_builder.getContext())">;

// A struct type for representing PCC Objects i.e. cache/directory/msg types
def StructType : DialectType<PCC_Dialect, CPred<"$_self.isa<StructType>()">,
    "struct", "::mlir::pcc::StructType">, BuildableType<"StructType::get($_builder.getContext())">;

def IntRangeType : DialectType<PCC_Dialect, CPred<"$_self.isa<IntRangeType>()">,
        "int_range", "::mlir::pcc::IntRangeType">, BuildableType<"IntRangeType::get($_builder.getContext())">;

def DataType : DialectType<PCC_Dialect, CPred<"$_self.isa<DataType>()">,
        "data", "::mlir::pcc::DataType">, BuildableType<"DataType::get($_builder.getContext())">;


#endif //PCC_TYPES