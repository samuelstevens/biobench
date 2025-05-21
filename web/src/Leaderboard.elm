module Leaderboard exposing (..)

import Benchmark
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
    = Fetched (Result Http.Error Payload)
    | Sort SortKey


type Requested a e
    = Loading
    | Loaded a
    | Failed e


type alias Payload =
    { meta : Metadata
    , checkpoints : List Benchmark.Checkpoint
    , priorTasks : List Benchmark.Task
    , benchmarkTasks : List Benchmark.Task
    , scores : List Benchmark.Score
    , bests : List Benchmark.Best
    }


type alias Metadata =
    { schema : Int
    , generated : Time.Posix
    , commit : String
    , seed : Int
    , alpha : Float
    , nBootstraps : Int
    }


payloadDecoder : D.Decoder Payload
payloadDecoder =
    D.map6 Payload
        (D.field "meta" metadataDecoder)
        (D.field "models" (D.list Benchmark.checkpointDecoder))
        (D.field "prior_work_tasks" (D.list Benchmark.taskDecoder))
        (D.field "benchmark_tasks" (D.list Benchmark.taskDecoder))
        (D.field "results" (D.list Benchmark.scoreDecoder))
        (D.field "bests" (D.list Benchmark.bestDecoder))


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


type alias Model =
    { requestedPayload : Requested Payload String
    , sortKey : SortKey
    , sortOrder : SortOrder
    }


type SortKey
    = CheckpointDisplay
    | TaskName String


type SortOrder
    = Increasing
    | Decreasing


opposite : SortOrder -> SortOrder
opposite o =
    case o of
        Increasing ->
            Decreasing

        Decreasing ->
            Increasing


type alias Row =
    { checkpoint : Benchmark.Checkpoint
    , scores : Dict.Dict String Float
    , best : Set.Set String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( { requestedPayload = Loading
      , sortKey = TaskName "beluga"
      , sortOrder = Decreasing
      }
    , Http.get
        { url = "data/results.json"
        , expect = Http.expectJson Fetched payloadDecoder
        }
    )


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        Fetched result ->
            case result of
                Ok payload ->
                    ( { model | requestedPayload = Loaded payload }, Cmd.none )

                Err err ->
                    ( { model
                        | requestedPayload = Failed (explainHttpError err)
                      }
                    , Cmd.none
                    )

        Sort key ->
            case ( model.sortKey, key ) of
                ( CheckpointDisplay, CheckpointDisplay ) ->
                    ( { model
                        | sortOrder = opposite model.sortOrder
                      }
                    , Cmd.none
                    )

                ( TaskName old, TaskName new ) ->
                    if old == new then
                        ( { model
                            | sortOrder = opposite model.sortOrder
                          }
                        , Cmd.none
                        )

                    else
                        ( { model | sortKey = key }, Cmd.none )

                ( _, _ ) ->
                    ( { model | sortKey = key }, Cmd.none )


view : Model -> Html.Html Msg
view model =
    case model.requestedPayload of
        Loading ->
            Html.div [] [ Html.text "Loading..." ]

        Failed err ->
            Html.div [] [ Html.text ("Failed: " ++ err) ]

        Loaded payload ->
            viewTable payload model.sortKey model.sortOrder


viewTable : Payload -> SortKey -> SortOrder -> Html.Html Msg
viewTable payload key order =
    let
        rows =
            pivotPayload payload
    in
    Html.table [ class "" ]
        [ Html.thead [ class "border-t border-b" ]
            [ Html.tr []
                ([ Html.th
                    [ class "text-left font-medium", Html.Events.onClick (Sort CheckpointDisplay) ]
                    [ Html.text "Checkpoint" ]
                 ]
                    ++ List.map
                        (viewTaskHeader key order)
                        payload.benchmarkTasks
                )
            ]
        , Html.tbody
            [ class "border-b" ]
            (rows |> sortRows key order |> List.map (viewRow key))
        ]


sortRows : SortKey -> SortOrder -> List Row -> List Row
sortRows key order rows =
    let
        ordered =
            case key of
                CheckpointDisplay ->
                    List.sortBy (.checkpoint >> .display) rows

                TaskName t ->
                    List.sortBy (.scores >> Dict.get t >> Maybe.withDefault (-1 / 0)) rows
    in
    case order of
        Increasing ->
            ordered

        Decreasing ->
            List.reverse ordered


viewTaskHeader : SortKey -> SortOrder -> Benchmark.Task -> Html.Html Msg
viewTaskHeader key order task =
    let
        suffix =
            case ( key, order ) of
                ( CheckpointDisplay, _ ) ->
                    ""

                ( TaskName name, Increasing ) ->
                    if name == task.name then
                        downArrow

                    else
                        ""

                ( TaskName name, Decreasing ) ->
                    if name == task.name then
                        upArrow

                    else
                        ""
    in
    Html.th
        [ class "text-right pl-2", Html.Events.onClick (Sort <| TaskName task.name) ]
        [ Html.text (task.display ++ suffix) ]


viewRow : SortKey -> Row -> Html.Html Msg
viewRow highlight row =
    let
        scores =
            row.scores
                |> Dict.toList
                |> List.sortBy (\pair -> Tuple.first pair)

        bests =
            scores
                |> List.map Tuple.first
                |> List.map (\t -> Set.member t row.best)
    in
    Html.tr
        [ class "hover:bg-biobench-cream-500" ]
        (Html.td
            [ class "text-left" ]
            [ Html.text row.checkpoint.display ]
            :: List.map2 (viewScoreCell highlight) bests scores
        )


viewScoreCell : SortKey -> Bool -> ( String, Float ) -> Html.Html Msg
viewScoreCell key best ( task, score ) =
    let
        highlight =
            case key of
                TaskName name ->
                    if name == task then
                        " italic"

                    else
                        ""

                _ ->
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


pivotPayload : Payload -> List Row
pivotPayload payload =
    List.map (pivotModelRow payload) payload.checkpoints


pivotModelRow : Payload -> Benchmark.Checkpoint -> Row
pivotModelRow payload checkpoint =
    let
        scores =
            payload.benchmarkTasks
                |> List.map .name
                |> List.map (findResult payload.scores checkpoint)
                |> List.map (Maybe.withDefault (-1 / 0))

        names =
            payload.benchmarkTasks
                |> List.map .name

        scoreDict =
            List.map2 Tuple.pair names scores
                |> Dict.fromList

        bests =
            payload.bests
                |> List.filter (.ties >> Set.member checkpoint.checkpoint)
                |> List.map .task
                |> Set.fromList
    in
    { checkpoint = checkpoint
    , scores = scoreDict
    , best = bests
    }



-- belongs : SortKey -> String -> Bool
-- belongs


findResult : List Benchmark.Score -> Benchmark.Checkpoint -> String -> Maybe Float
findResult scores checkpoint task =
    scores
        |> List.filter (\result -> result.task == task && result.checkpoint == checkpoint.checkpoint)
        |> List.map .mean
        |> List.head


downArrow : String
downArrow =
    String.fromChar (Char.fromCode 9650)


upArrow : String
upArrow =
    String.fromChar (Char.fromCode 9660)
