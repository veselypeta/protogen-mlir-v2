#ifndef PCC_OPS
#define PCC_OPS

def FooOp : PCC_Op<"foo"> {
    let summary = "foo op";
    let description = [{}];
    let results = (outs PCCType);
}

#endif // PCC_OPS