#ifndef PCC_PATTERNS
#define PCC_PATTERNS

include "PCC/PCC.td"

def createTransientState : NativeCodeCall<"createTransientState($_builder)">;
def : Pat<(InlineConstOp $theAttr), (createTransientState)>;

#endif // PCC_PATTERNS