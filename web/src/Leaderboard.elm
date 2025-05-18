module Leaderboard exposing (..)

import Browser
import Html
import Html.Attributes exposing (class)
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


type Requested a e
    = Loading
    | Loaded a
    | Failed e


type alias Payload =
    { meta : Metadata
    , models : List BenchmarkModel
    , priorTasks : List BenchmarkTask
    , benchmarkTasks : List BenchmarkTask
    , results : List BenchmarkResult
    , bests : List BenchmarkBest
    }


type alias Metadata =
    { schema : Int
    , generated : Time.Posix
    , commit : String
    , seed : Int
    , alpha : Float
    , nBootstraps : Int
    }


type alias BenchmarkModel =
    { checkpoint : String
    , display : String
    , family : String
    }


type alias BenchmarkTask =
    { name : String
    , display : String
    }


type alias BenchmarkResult =
    { task : String
    , model : String
    , mean : Float
    , meanBootstrap : Float
    , low : Float
    , high : Float
    }


type alias BenchmarkBest =
    { task : String
    , best : String
    , ties : Set.Set String
    }


payloadDecoder : D.Decoder Payload
payloadDecoder =
    D.map6 Payload
        (D.field "meta" metadataDecoder)
        (D.field "models" (D.list benchmarkModelDecoder))
        (D.field "prior_work_tasks" (D.list benchmarkTaskDecoder))
        (D.field "benchmark_tasks" (D.list benchmarkTaskDecoder))
        (D.field "results" (D.list benchmarkResultDecoder))
        (D.field "bests" (D.list benchmarkBestDecoder))


metadataDecoder : D.Decoder Metadata
metadataDecoder =
    D.map6 Metadata
        (D.field "schema" D.int)
        (D.field "generated" (D.map Time.millisToPosix D.int))
        (D.field "git_commit" D.string)
        (D.field "seed" D.int)
        (D.field "alpha" D.float)
        (D.field "n_bootstraps" D.int)


benchmarkResultDecoder : D.Decoder BenchmarkResult
benchmarkResultDecoder =
    D.map6 BenchmarkResult
        (D.field "task" D.string)
        (D.field "model" D.string)
        (D.field "mean" D.float)
        (D.field "bootstrap_mean" D.float)
        (D.field "ci_low" D.float)
        (D.field "ci_high" D.float)


benchmarkTaskDecoder : D.Decoder BenchmarkTask
benchmarkTaskDecoder =
    D.map2 BenchmarkTask
        (D.field "name" D.string)
        (D.field "display" D.string)


benchmarkModelDecoder : D.Decoder BenchmarkModel
benchmarkModelDecoder =
    D.map3 BenchmarkModel
        (D.field "ckpt" D.string)
        (D.field "display" D.string)
        (D.field "family" D.string)


benchmarkBestDecoder : D.Decoder BenchmarkBest
benchmarkBestDecoder =
    D.map3 BenchmarkBest
        (D.field "task" D.string)
        (D.field "best" D.string)
        (D.field "ties"
            (D.map Set.fromList (D.list D.string))
        )


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
    { requestedPayload : Requested Payload String }


init : () -> ( Model, Cmd Msg )
init _ =
    ( { requestedPayload = Loading }
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


view : Model -> Html.Html Msg
view model =
    case model.requestedPayload of
        Loading ->
            Html.div [] [ Html.text "Loading..." ]

        Failed err ->
            Html.div [] [ Html.text ("Failed: " ++ err) ]

        Loaded payload ->
            viewTable payload


viewTable : Payload -> Html.Html Msg
viewTable payload =
    Html.table [ class "w-full text-sm" ]
        [ Html.thead []
            [ Html.tr []
                (List.map viewTheadText [ "Task", "Model", "Mean", "CI Low", "CI High" ])
            ]
        , Html.tbody []
            (List.map viewRow payload.results)
        ]


viewTheadText : String -> Html.Html msg
viewTheadText s =
    Html.th [ class "text-left font-medium px-2 py-1 border-b" ] [ Html.text s ]


viewRow : BenchmarkResult -> Html.Html Msg
viewRow r =
    Html.tr [ class "hover:bg-gray-100" ]
        [ Html.td [ class "px-2 py-1" ] [ Html.text r.task ]
        , Html.td [ class "px-2 py-1" ] [ Html.text r.model ]
        , Html.td [ class "px-2 py-1 tabular-nums" ] [ Html.text (roundF r.mean) ]
        , Html.td [ class "px-2 py-1 tabular-nums" ] [ Html.text (roundF r.low) ]
        , Html.td [ class "px-2 py-1 tabular-nums" ] [ Html.text (roundF r.high) ]
        ]


roundF : Float -> String
roundF f =
    String.fromFloat (toFloat (round (f * 1000)) / 1000)
