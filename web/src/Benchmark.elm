module Benchmark exposing (Best, Checkpoint, Score, Task, bestDecoder, checkpointDecoder, scoreDecoder, taskDecoder)

import Json.Decode as D
import Set


type alias Checkpoint =
    { checkpoint : String
    , display : String
    , family : String
    }


checkpointDecoder : D.Decoder Checkpoint
checkpointDecoder =
    D.map3 Checkpoint
        (D.field "ckpt" D.string)
        (D.field "display" D.string)
        (D.field "family" D.string)


type alias Task =
    { name : String
    , display : String
    }


taskDecoder : D.Decoder Task
taskDecoder =
    D.map2 Task
        (D.field "name" D.string)
        (D.field "display" D.string)


type alias Score =
    { task : String
    , checkpoint : String
    , mean : Float
    , bootstrapMean : Float
    , low : Float
    , high : Float
    }


scoreDecoder : D.Decoder Score
scoreDecoder =
    D.map6 Score
        (D.field "task" D.string)
        (D.field "model" D.string)
        (D.field "mean" D.float)
        (D.field "bootstrap_mean" D.float)
        (D.field "ci_low" D.float)
        (D.field "ci_high" D.float)


type alias Best =
    { task : String
    , best : String
    , ties : Set.Set String
    }


bestDecoder : D.Decoder Best
bestDecoder =
    D.map3 Best
        (D.field "task" D.string)
        (D.field "best" D.string)
        (D.field "ties"
            (D.map Set.fromList (D.list D.string))
        )
