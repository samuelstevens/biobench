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
    | ToggleCol String


type Requested a e
    = Loading
    | Loaded a
    | Failed e


type alias Model =
    { requestedTable : Requested Table Http.Error

    -- Pickers
    , selectedCols : Set.Set String
    , selectedFamilies : Set.Set String
    , paramCountRange : ( Int, Int )

    -- Sorting
    , sortKey : String
    , sortOrder : Order
    }


type alias Table =
    { rows : List TableRow
    , cols : List TableCol
    , metadata : Metadata
    }


type alias TableRow =
    { checkpoint : Checkpoint
    , scores : Dict.Dict String Float
    , winners : Set.Set String
    }


type alias Checkpoint =
    { name : String
    , display : String
    , family : String
    , params : Int
    , release : Time.Posix
    }


type SortType
    = SortNumeric (Set.Set String -> TableRow -> Maybe Float)
    | SortString (Set.Set String -> TableRow -> Maybe String)
    | NotSortable


type alias TableCol =
    { key : String
    , display : String

    -- How to get the cell value
    -- (results in a class string and an Html.text string)
    , format : Set.Set String -> TableRow -> ( String, String )

    -- Information for SORTING
    , sortType : SortType
    }


type Order
    = Ascending
    | Descending


opposite : Order -> Order
opposite order =
    case order of
        Ascending ->
            Descending

        Descending ->
            Ascending


type alias Metadata =
    { schema : Int
    , generated : Time.Posix
    , commit : String
    , seed : Int
    , alpha : Float
    , nBootstraps : Int
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( { requestedTable = Loading
      , selectedCols = Set.empty

      -- TODO: implement
      , paramCountRange = ( 0, 10 ^ 12 )

      -- TODO: implement
      , selectedFamilies = Set.empty
      , sortKey = "mean"
      , sortOrder = Descending
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
                    ( { model
                        | requestedTable = Loaded table
                        , selectedCols =
                            table.cols
                                |> List.map .key
                                |> Set.fromList
                      }
                    , Cmd.none
                    )

                Err err ->
                    ( { model | requestedTable = Failed err }, Cmd.none )

        Sort key ->
            if model.sortKey == key then
                ( { model | sortOrder = opposite model.sortOrder }, Cmd.none )

            else
                ( { model | sortKey = key }, Cmd.none )

        ToggleCol key ->
            if Set.member key model.selectedCols then
                ( { model | selectedCols = Set.remove key model.selectedCols }, Cmd.none )

            else
                ( { model | selectedCols = Set.insert key model.selectedCols }, Cmd.none )


view : Model -> Html.Html Msg
view model =
    case model.requestedTable of
        Loading ->
            Html.div [] [ Html.text "Loading..." ]

        Failed err ->
            Html.div [] [ Html.text ("Failed: " ++ explainHttpError err) ]

        Loaded table ->
            Html.div []
                [ viewPicker model.selectedCols table
                , Html.table
                    [ class "w-full" ]
                    [ viewThead model.selectedCols model.sortKey model.sortOrder table
                    , viewTbody model.selectedCols model.sortKey model.sortOrder table
                    ]
                ]


viewPicker : Set.Set String -> Table -> Html.Html Msg
viewPicker selectedCols table =
    Html.div
        []
        (List.map
            (\col ->
                viewColCheckbox
                    (Set.member col.key selectedCols)
                    col
            )
            table.cols
        )


viewColCheckbox : Bool -> TableCol -> Html.Html Msg
viewColCheckbox checked col =
    -- TODO: Style
    Html.div
        [ class "cursor-pointer"
        , Html.Events.onClick (ToggleCol col.key)
        ]
        [ Html.input
            [ Html.Attributes.type_ "checkbox"
            , Html.Attributes.checked checked
            , class "cursor-pointer"
            ]
            []
        , Html.label
            [ class "cursor-pointer" ]
            [ Html.text col.display ]
        ]


viewThead : Set.Set String -> String -> Order -> Table -> Html.Html Msg
viewThead selectedCols sortKey sortOrder table =
    Html.thead
        [ class "border-t border-b py-1" ]
        (table.cols
            |> List.filter (\col -> Set.member col.key selectedCols)
            |> List.map (viewTh sortKey sortOrder)
        )


viewTh : String -> Order -> TableCol -> Html.Html Msg
viewTh sortKey sortOrder col =
    let
        extra =
            if sortKey == col.key then
                case sortOrder of
                    Descending ->
                        downArrow

                    Ascending ->
                        upArrow

            else
                ""
    in
    Html.th
        [ class "px-2", Html.Events.onClick (Sort col.key) ]
        [ Html.text (col.display ++ extra) ]


viewTbody : Set.Set String -> String -> Order -> Table -> Html.Html Msg
viewTbody selectedCols sortKey sortOrder table =
    let
        sortType =
            table.cols
                |> List.filter (\col -> col.key == sortKey)
                |> List.head
                |> Maybe.map .sortType
                |> Maybe.withDefault NotSortable

        sorted =
            case sortType of
                SortNumeric fn ->
                    List.sortBy (fn selectedCols >> Maybe.withDefault (-1 / 0)) table.rows

                SortString fn ->
                    List.sortBy (fn selectedCols >> Maybe.withDefault maxString) table.rows

                NotSortable ->
                    table.rows

        ordered =
            case sortOrder of
                Ascending ->
                    sorted

                Descending ->
                    List.reverse sorted
    in
    Html.tbody
        [ class "border-b" ]
        (List.map (viewTr selectedCols table.cols) ordered)


viewTr : Set.Set String -> List TableCol -> TableRow -> Html.Html Msg
viewTr selectedCols allCols row =
    let
        cols =
            allCols
                |> List.filter (\col -> Set.member col.key selectedCols)
    in
    Html.tr
        [ class "py-1" ]
        (List.map (viewTd selectedCols row) cols)


viewTd : Set.Set String -> TableRow -> TableCol -> Html.Html Msg
viewTd selectedCols row col =
    let
        ( cls, text ) =
            col.format selectedCols row
    in
    Html.td
        [ class ("px-2 " ++ cls) ]
        [ Html.text text ]



-- (List.map viewTd row


viewScore : Maybe Float -> String
viewScore score =
    case score of
        Nothing ->
            "-"

        Just val ->
            val * 100 |> Round.round 1


getBenchmarkScore : String -> Set.Set String -> TableRow -> Maybe Float
getBenchmarkScore task visibleKeys row =
    Dict.get task row.scores


viewBenchmarkScore : String -> Set.Set String -> TableRow -> ( String, String )
viewBenchmarkScore task visibleKeys row =
    ( "text-right font-mono"
    , getBenchmarkScore task visibleKeys row |> viewScore
    )


getMeanScore : Set.Set String -> Set.Set String -> TableRow -> Maybe Float
getMeanScore tasks selectedCols row =
    let
        selectedScores =
            tasks
                |> Set.intersect selectedCols
                |> Set.toList
                |> List.filterMap (\key -> Dict.get key row.scores)
    in
    -- selectedCols can be more than 9 (includes imagenet1k, newt, checkpoint, mean, etc).
    -- I think I might have to hardcode the benchmark tasks somewhere.
    if Debug.log "length" (List.length selectedScores) == Debug.log "size" (tasks |> Set.intersect selectedCols |> Set.size) then
        mean selectedScores

    else
        Nothing


viewMeanScore : Set.Set String -> Set.Set String -> TableRow -> ( String, String )
viewMeanScore tasks selectedCols row =
    ( "text-right font-mono"
    , getMeanScore tasks selectedCols row |> viewScore
    )


getCheckpoint : Set.Set String -> TableRow -> Maybe String
getCheckpoint cols row =
    Just row.checkpoint.display


viewCheckpoint : Set.Set String -> TableRow -> ( String, String )
viewCheckpoint cols row =
    ( "text-left"
    , row.checkpoint.display
    )


mean : List Float -> Maybe Float
mean xs =
    case xs of
        [] ->
            Nothing

        _ ->
            Just (List.sum xs / toFloat (List.length xs))



-- CONSTANTS


upArrow : String
upArrow =
    String.fromChar (Char.fromCode 9650)


downArrow : String
downArrow =
    String.fromChar (Char.fromCode 9660)


maxString : String
maxString =
    String.repeat 50 "\u{10FFFF}"



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
        -- [ CheckpointCell { name = "dummy", display = "dummy" }
        -- , { key = "params", display = "Params (M)" }
        -- , { key = "date", display = "Release" }
        -- , { key = Imagenet1kColumn, display = "ImageNet-1K" }
        -- , { key = NewtColumn, display = "NeWT" }
        -- , { key = MeanColumn, display = "Mean" }
        -- ]
        dynamicCols =
            List.map
                (\task ->
                    { key = task.name
                    , display = task.display
                    , format = viewBenchmarkScore task.name
                    , sortType = SortNumeric (getBenchmarkScore task.name)
                    }
                )
                payload.benchmarkTasks

        tasks =
            payload.benchmarkTasks |> List.map .name |> Set.fromList

        fixedCols =
            [ { key = "checkpoint", display = "Checkpoint", format = viewCheckpoint, sortType = SortString getCheckpoint }

            -- TODO: params, release date, model family
            , { key = "imagenet1k", display = "Imagenet-1K", format = viewBenchmarkScore "imagenet1k", sortType = SortNumeric (getBenchmarkScore "imagenet1k") }
            , { key = "newt", display = "NeWT", format = viewBenchmarkScore "newt", sortType = SortNumeric (getBenchmarkScore "newt") }
            , { key = "mean", display = "Mean", format = viewMeanScore tasks, sortType = SortNumeric (getMeanScore tasks) }
            ]

        cols =
            fixedCols ++ dynamicCols

        rows =
            List.map (makeRow payload) payload.checkpoints
    in
    { cols = cols, rows = rows, metadata = payload.metadata }


makeRow : Payload -> Checkpoint -> TableRow
makeRow payload checkpoint =
    let
        scores =
            getScores checkpoint payload.scores
    in
    { checkpoint = checkpoint
    , scores = scores
    , winners = getWinners checkpoint payload.bests
    }


getScore : Checkpoint -> String -> List Score -> Maybe Float
getScore checkpoint task scores =
    scores
        |> List.filter (\score -> score.task == task && score.checkpoint == checkpoint.name)
        |> List.map .mean
        |> List.head


getScores : Checkpoint -> List Score -> Dict.Dict String Float
getScores checkpoint scores =
    scores
        |> List.filter (\score -> score.checkpoint == checkpoint.name)
        |> List.map (\score -> ( score.task, score.mean ))
        |> Dict.fromList


getWinners : Checkpoint -> List Best -> Set.Set String
getWinners checkpoint bests =
    bests
        |> List.filter (\best -> Set.member checkpoint.name best.ties)
        |> List.map .task
        |> Set.fromList


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
    , ties : Set.Set String
    }


bestDecoder : D.Decoder Best
bestDecoder =
    D.map2 Best
        (D.field "task" D.string)
        (D.field "ties"
            (D.map Set.fromList (D.list D.string))
        )
