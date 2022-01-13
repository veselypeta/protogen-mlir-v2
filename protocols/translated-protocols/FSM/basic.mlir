{

fsm.machine @cache(){
    %val = fsm.variable "val" {initValue = 21} : i64

    fsm.state @S {

        fsm.transition @store {
            send GetM to directory
        }

        fsm.transition @Fwd_GetS{
            transition to I
        }

    }


    fsm.state "S_store" {links=[@S::@store]} {



        // We can easily find racing transactions that can occur by inspecting
        // which messages can arrive in the logical start state
        // Can maybe also be done with links??
        fsm.transition "Fwd_GetS" {

            // Through the links on the state we can tell which state we transitioned here to
            // and which message was sent to the directory
            // PROBLEM :    what if there are multiple links?
            //              will a transient state always only have one link?

            // Logically we must transition to a state that represents starting from the new start state
            // but having sent that message

            // If such a case exists then we use it?

            // We then find the transition in the directory which sent us the message
            // and determine in which state the directory will be in when it receives our message

            /// The BIG Question is?
            // What state do we transition to?
            // Do we have create a new transient state?
            // If so what does this new transient state do?


            // From the racing handler (S , Fwd_GetS, I)
            // we should transition to a state that mimics: Starting from state I & sent a GetM
            // What if this is not a valid message to send in this state?
            // Directory can UPGRADE messages or ignore them?


            // Simple case:
            // The state already exists -> we transition to it and we are done.

            // The state does not exist:
            // How will the directory react? or How will we make the directory react?

            // Since the directory has sent us a message we can see which state the directory is in
            // and in which state it will be when it sent up the message

            // D_end is the end state of the directory after sending us our message
            // if (D_end, our_msg, some_state) is handled
            // then we must check what is does and do the right thing.

            // What is the right thing?

            // Suppose that:
            // cache (A, fwd_action, ??)
            // directory(D, our_message, E) { send ack message }

            // How do we know if we expect a message or not?
            // is it unknowable?


            // FULL EXAMPLE

            // In the MI case
            // M_evict is a transient state which can receive a Fwd_GetM message from the directory
            // The original sent message was a PutM
            // the Fwd_GetM transaction then transitions to final state I

            // So we need to transition to a state which represents starting from state I and having sent a PutM
            // This state does not exist since we don't sent PutM messages in state I

            // What state is the directory in?
            // The directory sent a Fwd_GetM from state M and transitioned to state M
            // in full --> (M, GetM, M) { send Fwd_GetM to owner }

            // The directory remains in state M when our PutM arrives
            // How is this handled?
            // The directory always sends a PutAck, but only transitions to I iff the PutAck was sent by the owner

            // In this case we have to create a new state to accept the PutAck from the directory
            // after which we can transition to I
            // But is this always knowable?



            // MSI Example

            // lets try and optimize the S -> S_store -> S_store_ack -> M

            // In state S_store we have sent an Upgrade message to the directory
            // We also know that we can receive an Inv racing message, which ends in state I

            // The directory sends an Inv message from (S, Upgrade, M)
            // So the directory will receive Upgrade in state M.
            // this is unhandled by the directory we can choose what to do.
            // choice 1 : do nothing
            // we short-circuit and go directly to state I.

            // So we introduce the transition (S_store, Inv, I) { send Inv_Ack }

            // state S_store_ack is now actually in logical start state S still
            // But the directory will send messages to us for state M, but we have
            // not got there yet.
            // In this state we can however no longer receive Inv messages
            // since the directory is not in state S
            // However there is no harm in including this case so we can do it.










        }

    }
}


fsm.machine @directory(){

    fsm.state @S {
        fsm.transition @GetS {dest=@S}{
            send Fwd_GetS
        }
    }

}

}



message @Resp{
    %cl = variable "cl" : Data
}




machine @cache{

    %state = variable -> State<I>
    %cl = variable -> Data

    state @S {

        transition @store {
            // send GetM to directory
            %GetM = message @Request "GetM" @cache @directory

        }

        transition @Inv (%inv : Msg<"Ack">) {
            // messages have a unique Type and a unique Name
            // i.e. the message Inv
            // has type -> Ack & name -> Inv

            // type will almost surely by a symbolic reference
            // name will almost surely be a string reference

            // All other inputs should be SSA values???
            // this means that we need a message input into transitions which expect them

            // we can have type checking for what fields are accessible

            %1 = ref @cache -> ID
            %2 = ref @directory -> ID
            %2 = access %inv {key="src"} -> ID
            %msg = message @Resp "Inv_Ack" %1 %2 %cl

        }
    }

    state @S_store {isTransient=true, prevTransition=@cache::@S::@store} {

        transition @GetM_Ack_D {
            transition to M
        }

        transition @GetM_Ack_AD {
            transition possibility

        }

        transition @Inv_Ack(%msg : struct<ID, ID, ID>) {

        }

    }

    state @S_store_GetM_Ack_AD {isTransient=true, prevState=@cache::@S_store} {

    }



}