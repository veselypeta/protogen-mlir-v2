#ifndef PCC_OPS
#define PCC_OPS

include "PCCDialect.td"

def FooOp : PCC_Op<"foo"> {
    let summary = "foo op";
    let description = [{}];
}

#endif // PCC_OPS