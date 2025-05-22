module Leaderboard exposing (..)

import Browser
import Dict
import Html
import Html.Attributes exposing (class)
import Html.Events
import Http
import Json.Decode as D
import Round
import Set
import Time


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = \_ -> Sub.none
        , view = view
        }


type Msg
    = Fetched (Result Http.Error Table)
    | Sort String


type Requested a e
    = Loading
    | Loaded a
    | Failed e


type alias Table =
    { cols : List Column
    , rows : List Row
    , metadata : Metadata
    }


type alias Column =
    { name : String
    , display : String
    }


type alias Row =
    { checkpoint : Checkpoint
    , imagenet1k : Float
    , newt : Float
    , mean : Float
    , scores : Dict.Dict String Float
    , sota : Set.Set String
    }


type alias Metadata =
    { schema : Int
    , generated : Time.Posix
    , commit : String
    , seed : Int
    , alpha : Float
    , nBootstraps : Int
    }


type alias Model =
    { requestedTable : Requested Table String
    , sortKey : String
    , sortDecreasing : Bool
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( { requestedTable = Loading
      , sortKey = "mean"
      , sortDecreasing = True
      }
    , Http.get
        { url = "data/results.json"
        , expect = Http.expectJson Fetched tableDecoder
        }
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Fetched result ->
            case result of
                Ok table ->
                    ( { model | requestedTable = Loaded table }, Cmd.none )

                Err err ->
                    ( { model
                        | requestedTable = Failed (explainHttpError err)
                      }
                    , Cmd.none
                    )

        Sort key ->
            if model.sortKey == key then
                ( { model
                    | sortDecreasing = not model.sortDecreasing
                  }
                , Cmd.none
                )

            else
                ( { model | sortKey = key }, Cmd.none )


view : Model -> Html.Html Msg
view model =
    case model.requestedTable of
        Loading ->
            Html.div [] [ Html.text "Loading..." ]

        Failed err ->
            Html.div [] [ Html.text ("Failed: " ++ err) ]

        Loaded table ->
            viewTable table model.sortKey model.sortDecreasing


viewTable : Table -> String -> Bool -> Html.Html Msg
viewTable table key decreasing =
    Html.table
        [ class "" ]
        [ Html.thead
            [ class "border-t border-b" ]
            [ Html.tr
                []
                (List.map (viewHeaderCell key decreasing) table.cols)
            ]
        , Html.tbody
            [ class "border-b" ]
            (table.rows
                |> orderedBy key decreasing
                |> List.map (viewTableRow key)
            )
        ]


orderedBy : String -> Bool -> List Row -> List Row
orderedBy key decreasing rows =
    let
        ordered =
            case key of
                "checkpoint" ->
                    List.sortBy (.checkpoint >> .display) rows

                "params" ->
                    List.sortBy (.checkpoint >> .params) rows

                "release" ->
                    List.sortBy (.checkpoint >> .release >> Time.posixToMillis) rows

                "imagenet1k" ->
                    List.sortBy .imagenet1k rows

                "newt" ->
                    List.sortBy .newt rows

                "mean" ->
                    List.sortBy .mean rows

                name ->
                    List.sortBy (.scores >> Dict.get name >> Maybe.withDefault (-1 / 0)) rows
    in
    if decreasing then
        List.reverse ordered

    else
        ordered


viewHeaderCell : String -> Bool -> Column -> Html.Html Msg
viewHeaderCell key decreasing col =
    let
        suffix =
            if key == col.name then
                if decreasing then
                    upArrow

                else
                    downArrow

            else
                ""
    in
    Html.th
        [ class "text-right pl-2", Html.Events.onClick (Sort col.name) ]
        [ Html.text (col.display ++ suffix) ]


viewTableRow : String -> Row -> Html.Html Msg
viewTableRow key row =
    let
        scores =
            row.scores
                |> Dict.toList
                |> List.sortBy (\pair -> Tuple.first pair)

        bests =
            scores
                |> List.map Tuple.first
                |> List.map (\t -> Set.member t row.sota)

        benchmarkCells =
            List.map2 (viewScoreCell key) bests scores
    in
    Html.tr
        [ class "hover:bg-biobench-cream-500" ]
        ([ Html.td
            [ class "text-left" ]
            [ Html.text row.checkpoint.display ]

         -- , Html.td
         --    [ class "text-right" ]
         --    [ Html.text (String.fromInt row.checkpoint.params) ]
         -- , Html.td
         --    [ class "text-right" ]
         --    [ Html.text "TODO" ]
         , viewScoreCell key (Set.member "imagenet1k" row.sota) ( "imagenet1k", row.imagenet1k )
         , viewScoreCell key (Set.member "newt" row.sota) ( "newt", row.newt )
         , viewScoreCell key (Set.member "mean" row.sota) ( "mean", row.mean )
         ]
            ++ benchmarkCells
        )


viewScoreCell : String -> Bool -> ( String, Float ) -> Html.Html Msg
viewScoreCell key best ( task, score ) =
    let
        highlight =
            if key == task then
                " italic"

            else
                ""

        bold =
            if best then
                " font-bold"

            else
                ""
    in
    Html.td
        [ class ("text-right font-mono" ++ highlight ++ bold) ]
        [ Html.text (viewScore score) ]


viewScore : Float -> String
viewScore score =
    if score < 0 then
        "-"

    else
        score * 100 |> Round.round 1



-- CONSTANTS


downArrow : String
downArrow =
    String.fromChar (Char.fromCode 9650)


upArrow : String
upArrow =
    String.fromChar (Char.fromCode 9660)



-- HTTP API


tableDecoder : D.Decoder Table
tableDecoder =
    D.map pivotPayload payloadDecoder


type alias Payload =
    { metadata : Metadata
    , checkpoints : List Checkpoint
    , priorTasks : List Task
    , benchmarkTasks : List Task
    , scores : List Score
    , bests : List Best
    }


payloadDecoder : D.Decoder Payload
payloadDecoder =
    D.map6 Payload
        (D.field "meta" metadataDecoder)
        (D.field "models" (D.list checkpointDecoder))
        (D.field "prior_work_tasks" (D.list taskDecoder))
        (D.field "benchmark_tasks" (D.list taskDecoder))
        (D.field "results" (D.list scoreDecoder))
        (D.field "bests" (D.list bestDecoder))


metadataDecoder : D.Decoder Metadata
metadataDecoder =
    D.map6 Metadata
        (D.field "schema" D.int)
        (D.field "generated" (D.map Time.millisToPosix D.int))
        (D.field "git_commit" D.string)
        (D.field "seed" D.int)
        (D.field "alpha" D.float)
        (D.field "n_bootstraps" D.int)


explainHttpError : Http.Error -> String
explainHttpError err =
    case err of
        Http.BadUrl url ->
            "Invalid URL: " ++ url

        Http.Timeout ->
            "Request timed out."

        Http.NetworkError ->
            "Unknown network error."

        Http.BadStatus status ->
            "Got status code: " ++ String.fromInt status

        Http.BadBody msg ->
            "Bad body: " ++ msg


pivotPayload : Payload -> Table
pivotPayload payload =
    let
        cols =
            [ { name = "checkpoint", display = "Checkpoint" }

            -- , { name = "params", display = "Params (M)" }
            -- , { name = "date", display = "Release" }
            , { name = "imagenet1k", display = "ImageNet-1K" }
            , { name = "newt", display = "NeWT" }
            , { name = "mean", display = "Mean" }
            ]
                ++ payload.benchmarkTasks

        rows =
            List.map (makeRow payload) payload.checkpoints
    in
    { cols = cols, rows = Debug.log "rows" rows, metadata = payload.metadata }


makeRow : Payload -> Checkpoint -> Row
makeRow payload checkpoint =
    let
        scores =
            getScores checkpoint payload.scores
    in
    { checkpoint = checkpoint
    , imagenet1k = getScore checkpoint "imagenet1k" payload.scores
    , newt = getScore checkpoint "newt" payload.scores
    , mean = scores |> Dict.toList |> List.map Tuple.second |> mean
    , scores = scores
    , sota = getSotas checkpoint payload.bests
    }


getScore : Checkpoint -> String -> List Score -> Float
getScore checkpoint task scores =
    scores
        |> List.filter (\score -> score.task == task && score.checkpoint == checkpoint.name)
        |> List.map .mean
        |> List.head
        |> Maybe.withDefault (-1 / 0)


getScores : Checkpoint -> List Score -> Dict.Dict String Float
getScores checkpoint scores =
    scores
        |> List.filter (\score -> score.task /= "imagenet1k" && score.task /= "newt" && score.checkpoint == checkpoint.name)
        |> List.map (\score -> ( score.task, score.mean ))
        |> Dict.fromList


getSotas : Checkpoint -> List Best -> Set.Set String
getSotas checkpoint bests =
    bests
        |> List.filter (\best -> Set.member checkpoint.name best.ties)
        |> List.map .task
        |> Set.fromList


mean : List Float -> Float
mean xs =
    case xs of
        [] ->
            0

        _ ->
            List.sum xs / toFloat (List.length xs)


type alias Checkpoint =
    { name : String
    , display : String
    , family : String
    , params : Int
    , release : Time.Posix
    }


checkpointDecoder : D.Decoder Checkpoint
checkpointDecoder =
    D.map5 Checkpoint
        (D.field "ckpt" D.string)
        (D.field "display" D.string)
        (D.field "family" D.string)
        (D.succeed 0)
        (D.succeed (Time.millisToPosix 0))


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
