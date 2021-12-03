module{

    "pcc.cache"() ({
        %0 = "pcc.variable" () : () -> i64
        %1 = "pcc.variable" () : () -> i64

        %net = "pcc.network" () : () -> i64


        "pcc.state"()({
            "pcc.transition"()({
                %msg = "pcc.message"(){m="GetM"} : () -> i64
                "pcc.send" (%msg, %net) : (i64, i64) -> ()
                %state = "pcc.constant" (){state="I_load"} : () -> i64
                "pcc.update" (%0, %state) : (i64, i64) -> ()
            }){action="load"} : () -> ()
        }){sym_name="I"} : () -> ()

        "pcc.state"()({
            ^entry(%arg: i64):
                "pcc.update"(%1, %arg) : (i64, i64) -> ()
        }) {action="GetM_AckD"}: () -> ()


    }): () -> ()

}