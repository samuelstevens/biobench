module Leaderboard exposing (..)

import Benchmark
import Browser
import Dict
import Html
import Html.Attributes exposing (class)
import Html.Events
import Http
import Json.Decode as D
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
    | ImageNet1K
    | Newt
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


init : () -> ( Model, Cmd Msg )
init _ =
    ( { requestedPayload = Loading
      , sortKey = ImageNet1K
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

                ( ImageNet1K, ImageNet1K ) ->
                    ( { model
                        | sortOrder = opposite model.sortOrder
                      }
                    , Cmd.none
                    )

                ( Newt, Newt ) ->
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
    Html.table [ class "w-full text-sm" ]
        [ Html.thead [ class "border-t border-b" ]
            [ Html.tr []
                ([ Html.th
                    [ class "text-left font-medium px-2 py-1", Html.Events.onClick (Sort CheckpointDisplay) ]
                    [ Html.text "Checkpoint" ]
                 , Html.th
                    [ class "text-left font-medium px-2 py-1", Html.Events.onClick (Sort ImageNet1K) ]
                    [ Html.text "ImageNet-1K" ]
                 , Html.th
                    [ class "text-left font-medium px-2 py-1", Html.Events.onClick (Sort Newt) ]
                    [ Html.text "NeWT" ]
                 ]
                    ++ List.map
                        (\task ->
                            Html.th
                                [ class "text-left font-medium px-2 py-1", Html.Events.onClick (Sort (TaskName task.name)) ]
                                [ Html.text task.display ]
                        )
                        payload.benchmarkTasks
                )
            ]
        , Html.tbody
            [ class "border-b" ]
            (rows |> sortRows key order |> List.map viewRow)
        ]


sortRows : SortKey -> SortOrder -> List Row -> List Row
sortRows key order rows =
    let
        ordered =
            case key of
                CheckpointDisplay ->
                    List.sortBy (.checkpoint >> .display) rows

                ImageNet1K ->
                    List.sortBy .imagenet1k rows

                Newt ->
                    List.sortBy .newt rows

                TaskName t ->
                    List.sortBy (.scores >> Dict.get t >> Maybe.withDefault (-1 / 0)) rows
    in
    case order of
        Increasing ->
            ordered

        Decreasing ->
            List.reverse ordered


type alias Row =
    { checkpoint : Benchmark.Checkpoint
    , imagenet1k : Float
    , newt : Float
    , scores : Dict.Dict String Float
    }


viewRow : Row -> Html.Html Msg
viewRow row =
    Html.tr [ class "hover:bg-biobench-cream-500" ]
        ([ Html.td [ class "px-2 py-1" ] [ Html.text row.checkpoint.display ]
         , Html.td [ class "px-2 py-1" ] [ Html.text (String.fromFloat row.imagenet1k) ]
         , Html.td [ class "px-2 py-1" ] [ Html.text (String.fromFloat row.newt) ]
         ]
            ++ (row.scores
                    |> Dict.toList
                    |> List.sortBy (\pair -> Tuple.first pair)
                    |> List.map
                        (\pair ->
                            Html.th [ class "px-2 py-1" ] [ Html.text (pair |> Tuple.second |> String.fromFloat) ]
                        )
               )
        )



-- (List.map (\txt -> Html.td [ class "px-2 py-1" ] [ Html.text txt ]) row)


pivotPayload : Payload -> List Row
pivotPayload payload =
    List.map (pivotModelRow payload) payload.checkpoints


pivotModelRow : Payload -> Benchmark.Checkpoint -> Row
pivotModelRow payload checkpoint =
    let
        imagenet1k =
            findResult payload.scores checkpoint "imagenet1k"
                |> Maybe.withDefault (-1 / 0)

        newt =
            findResult payload.scores checkpoint "newt"
                |> Maybe.withDefault (-1 / 0)

        others =
            payload.benchmarkTasks
                |> List.map .name
                |> List.map (findResult payload.scores checkpoint)
                |> List.map (Maybe.withDefault (-1 / 0))

        names =
            payload.benchmarkTasks
                |> List.map .name

        scoreDict =
            List.map2 Tuple.pair names others
                |> Dict.fromList
    in
    { checkpoint = checkpoint
    , imagenet1k = imagenet1k
    , newt = newt
    , scores = scoreDict
    }


findResult : List Benchmark.Score -> Benchmark.Checkpoint -> String -> Maybe Float
findResult scores checkpoint task =
    scores
        |> List.filter (\result -> result.task == task && result.checkpoint == checkpoint.checkpoint)
        |> List.map .mean
        |> List.head
