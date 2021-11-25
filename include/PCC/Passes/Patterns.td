#ifndef PCC_PATTERNS
#define PCC_PATTERNS

include "PCC/PCC.td"

def IncludesTransientState : Constraint<CPred<"$_self.hasNestedAwait()">, "nested await">;

def createTransientState : NativeCodeCall<"createTransientState($_builder)">;
def : Pat<(ProcessOp SymbolNameAttr:$symbol, TypeAttr:$type), (createTransientState)>;


#endif // PCC_PATTERNS