module {
    %0 = "pcc.foo" () : () -> !pcc.id
    %1 = "pcc.foo" () : () -> !pcc.network<ordered>
    %2 = "pcc.foo" () : () -> !pcc.network<unordered>
    %3 = "pcc.foo" () : () -> !pcc.state<I>
    %4 = "pcc.foo" () : () -> !pcc.state<M>
    %5 = "pcc.foo" () : () -> !pcc.set<!pcc.state<I>, 5>
    %6 = "pcc.foo" () : () -> !pcc.struct<!pcc.state<I>, !pcc.id, !pcc.network<ordered>>
}