#ifndef PCC_PATTERNS
#define PCC_PATTERNS

include "PCC/PCC.td"

def createTransientState : NativeCodeCall<"createTransientState($_builder)">;
def : Pat<(InlineConstOp $theAttr), (createTransientState)>;

// You could potentially implement a pattern such as
// if(true) { ... blah } => blah

#endif // PCC_PATTERNS