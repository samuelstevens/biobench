module Leaderboard exposing (..)

import Browser
import Chart as C
import Chart.Attributes as CA
import Dict
import Html exposing (Html)
import Html.Attributes exposing (class, style)
import Html.Events
import Html.Keyed
import Http
import Json.Decode as D
import Round
import Set
import Svg
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
    | ToggleFamily String
    | SetLayout Layout



-- | DragStart
-- | DragMove Float
-- | DragEnd


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

    -- UI
    , layout : Layout
    }


type Layout
    = TableOnly
    | ChartsOnly
    | Split Float -- Float = pct width 0–1


layoutEq : Layout -> Layout -> Bool
layoutEq a b =
    case ( a, b ) of
        ( TableOnly, TableOnly ) ->
            True

        ( ChartsOnly, ChartsOnly ) ->
            True

        ( Split _, Split _ ) ->
            True

        _ ->
            False


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
    , release : Maybe Time.Posix
    , params : Maybe Int
    , resolution : Maybe Int
    }


type SortType
    = SortNumeric (Set.Set String -> TableRow -> Maybe Float)
    | SortString (Set.Set String -> TableRow -> Maybe String)
    | NotSortable


getNumeric : SortType -> Maybe (Set.Set String -> TableRow -> Maybe Float)
getNumeric sortType =
    case sortType of
        SortNumeric fn ->
            Just fn

        _ ->
            Nothing


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
      , layout = Split 0.5
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
                        , selectedFamilies =
                            table.rows
                                |> List.map (.checkpoint >> .family)
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

        ToggleFamily key ->
            if Set.member key model.selectedFamilies then
                ( { model | selectedFamilies = Set.remove key model.selectedFamilies }, Cmd.none )

            else
                ( { model | selectedFamilies = Set.insert key model.selectedFamilies }, Cmd.none )

        SetLayout layout ->
            ( { model | layout = layout }, Cmd.none )


view : Model -> Html Msg
view model =
    case model.requestedTable of
        Loading ->
            Html.div [] [ Html.text "Loading..." ]

        Failed err ->
            Html.div [] [ Html.text ("Failed: " ++ explainHttpError err) ]

        Loaded table ->
            let
                tableContent =
                    viewTable model.selectedCols model.selectedFamilies model.sortKey model.sortOrder table

                chartContent =
                    viewCharts model.selectedCols model.selectedFamilies table
            in
            Html.div []
                [ viewPickers model.layout model.selectedCols model.selectedFamilies table
                , Html.div
                    [ class "flex" ]
                    (case model.layout of
                        TableOnly ->
                            [ viewTablePane tableContent 100 ]

                        ChartsOnly ->
                            [ viewChartPane chartContent 100 ]

                        Split pct ->
                            [ viewTablePane tableContent ((pct - 0.01) * 100)
                            , viewDragHandle
                            , viewChartPane chartContent (100 - (pct + 0.01) * 100)
                            ]
                    )
                ]


viewTablePane : Html Msg -> Float -> Html Msg
viewTablePane content w =
    Html.div
        [ style "width" (String.fromFloat w ++ "%")
        , class "overflow-x-auto"
        ]
        [ content ]


viewChartPane : Html Msg -> Float -> Html Msg
viewChartPane content w =
    Html.div
        [ style "width" (String.fromFloat w ++ "%")
        , class "overflow-y-auto"
        ]
        [ content ]


viewDragHandle =
    Html.div
        [ class "w-1 bg-black/10 hover:bg-gold cursor-col-resize select-none"

        -- , Html.Events.onMouseDown DragStart
        -- , Html.Events.on "mousemove" (Decode.map DragMove mousePos)
        -- , Html.Events.onMouseUp DragEnd
        ]
        []


viewPickers : Layout -> Set.Set String -> Set.Set String -> Table -> Html Msg
viewPickers layout selectedCols selectedFamilies table =
    let
        allFamilies =
            table.rows
                |> List.map (.checkpoint >> .family)
                |> Set.fromList
                |> Set.toList
                |> List.sort
    in
    Html.div
        [ class "flex flex-wrap gap-2" ]
        [ viewFieldset "Panes"
            [ viewLabeledRadio
                "table-only"
                (layoutEq layout TableOnly)
                (\_ -> SetLayout TableOnly)
                "Tables"
            , viewLabeledRadio
                "split"
                (layoutEq layout (Split -1))
                (\_ -> SetLayout (Split 0.5))
                "Both"
            , viewLabeledRadio
                "charts-only"
                (layoutEq layout ChartsOnly)
                (\_ -> SetLayout ChartsOnly)
                "Charts"
            ]
        , viewFieldset "Columns"
            (List.map
                (\col ->
                    viewColCheckbox
                        (Set.member col.key selectedCols)
                        col
                )
                table.cols
            )
        , viewFieldset "Model Families"
            (List.map
                (\family ->
                    viewFamilyCheckbox
                        (Set.member family selectedFamilies)
                        family
                )
                allFamilies
            )
        ]


viewColCheckbox : Bool -> TableCol -> Html Msg
viewColCheckbox checked col =
    viewLabeledCheckbox checked (\_ -> ToggleCol col.key) col.display


viewFamilyCheckbox : Bool -> String -> Html Msg
viewFamilyCheckbox checked family =
    viewLabeledCheckbox checked (\_ -> ToggleFamily family) family


viewTable : Set.Set String -> Set.Set String -> String -> Order -> Table -> Html Msg
viewTable selectedCols selectedFamilies sortKey sortOrder table =
    Html.table
        [ class "w-full text-xs sm:text-sm mt-2" ]
        [ viewThead selectedCols sortKey sortOrder table
        , viewTbody selectedCols selectedFamilies sortKey sortOrder table
        ]


viewThead : Set.Set String -> String -> Order -> Table -> Html Msg
viewThead selectedCols sortKey sortOrder table =
    Html.thead
        [ class "border-t border-b py-1" ]
        (table.cols
            |> List.filter (\col -> Set.member col.key selectedCols)
            |> List.map (viewTh sortKey sortOrder)
        )


viewTh : String -> Order -> TableCol -> Html Msg
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


viewTbody : Set.Set String -> Set.Set String -> String -> Order -> Table -> Html Msg
viewTbody selectedCols selectedFamilies sortKey sortOrder table =
    let
        filtered =
            table.rows
                |> List.filter (\row -> Set.member row.checkpoint.family selectedFamilies)

        sortType =
            table.cols
                |> List.filter (\col -> col.key == sortKey)
                |> List.head
                |> Maybe.map .sortType
                |> Maybe.withDefault NotSortable

        sorted =
            case sortType of
                SortNumeric fn ->
                    List.sortBy (fn selectedCols >> Maybe.withDefault (-1 / 0)) filtered

                SortString fn ->
                    List.sortBy (fn selectedCols >> Maybe.withDefault maxString) filtered

                NotSortable ->
                    filtered

        ordered =
            case sortOrder of
                Ascending ->
                    sorted

                Descending ->
                    List.reverse sorted
    in
    Html.Keyed.node "tbody"
        [ class "border-b" ]
        (List.map
            (\row ->
                ( row.checkpoint.name
                , viewTr selectedCols table.cols row
                )
            )
            ordered
        )


viewTr : Set.Set String -> List TableCol -> TableRow -> Html Msg
viewTr selectedCols allCols row =
    let
        cols =
            allCols
                |> List.filter (\col -> Set.member col.key selectedCols)
    in
    Html.tr
        [ class "py-1 hover:bg-biobench-cream/20 transition-colors" ]
        (List.map (viewTd selectedCols row) cols)


viewTd : Set.Set String -> TableRow -> TableCol -> Html Msg
viewTd selectedCols row col =
    let
        ( cls, text ) =
            col.format selectedCols row

        winner =
            if Set.member col.key row.winners then
                " font-bold"

            else
                ""
    in
    Html.td
        [ class ("px-2 " ++ cls ++ winner) ]
        [ Html.text text ]


viewCharts : Set.Set String -> Set.Set String -> Table -> Html Msg
viewCharts selectedCols selectedFamilies table =
    let
        rows =
            table.rows
                |> List.filter (\r -> Set.member r.checkpoint.family selectedFamilies)

        -- I want a list of getters (TableRow -> Maybe Float) and a list of titles (TableCol.display) AI!
        getters =
            table.cols
                |> List.filter (\c -> Set.member c.key selectedCols)
                |> List.filterMap (\c -> .sortType >> getNumeric)
                |> List.map (\fn -> fn selectedCols)
    in
    Html.div
        [ class "grid grid-cols-3 gap-2" ]
        (List.map2 (viewBarChart rows) getters)


viewBarChart : List TableRow -> String -> (TableRow -> Maybe Float) -> Html Msg
viewBarChart rows title getter =
    case List.filterMap (withFamily getter) rows of
        [] ->
            Html.div [ class "hidden" ] []

        data ->
            Html.div [ class "" ]
                [ C.chart
                    [ CA.height 200
                    , CA.width 200
                    , CA.margin
                        { top = 25
                        , bottom = 10
                        , left = 37
                        , right = 10
                        }
                    , CA.domain
                        [ CA.lowest 0 CA.exactly
                        , CA.highest 100 CA.exactly
                        ]
                    , CA.htmlAttrs
                        [ class "" ]
                    ]
                    [ C.yLabels [ CA.amount 3 ]
                    , C.yTicks [ CA.amount 3 ]
                    , C.bars
                        []
                        [ C.bar (.score >> (*) 100) []
                            |> C.variation
                                (\index datum ->
                                    [ CA.color (datum.family |> familyColor |> toCssColorVar) ]
                                )
                        ]
                        data
                    , C.labelAt
                        CA.middle
                        .max
                        [ CA.fontSize 14, CA.moveUp 12 ]
                        [ Svg.text title ]
                    ]
                ]


withFamily : (TableRow -> Maybe Float) -> TableRow -> Maybe { score : Float, family : String }
withFamily getter row =
    case getter row of
        Nothing ->
            Nothing

        Just score ->
            Just { score = score, family = row.checkpoint.family }


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
    let
        score =
            getBenchmarkScore task visibleKeys row
    in
    ( "text-right font-mono "
    , score |> viewScore
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
    if List.length selectedScores == (tasks |> Set.intersect selectedCols |> Set.size) then
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


getCheckpointParams : Set.Set String -> TableRow -> Maybe Float
getCheckpointParams _ row =
    row.checkpoint.params |> Maybe.map toFloat


viewCheckpointParams : Set.Set String -> TableRow -> ( String, String )
viewCheckpointParams selectedCols row =
    ( "text-left"
    , getCheckpointParams selectedCols row
        |> Maybe.map ((/) (10 ^ 6))
        |> Maybe.map (Round.round 1)
        |> Maybe.withDefault ""
    )


getCheckpointRelease : Set.Set String -> TableRow -> Maybe Float
getCheckpointRelease _ row =
    row.checkpoint.release |> Maybe.map (Time.posixToMillis >> toFloat)


viewCheckpointRelease : Set.Set String -> TableRow -> ( String, String )
viewCheckpointRelease _ row =
    ( "text-left"
    , row.checkpoint.release
        |> Maybe.map (Time.toYear Time.utc >> String.fromInt)
        |> Maybe.withDefault "unknown"
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


familyColor : String -> String
familyColor fam =
    -- Include all bg-VAR as comments to force tailwind to include all the colors as variables in the final CSS.
    -- This is an example of a comment changing system behavior!!
    case fam of
        "CLIP" ->
            -- bg-biobench-blue
            "biobench-blue"

        "SigLIP" ->
            -- bg-biobench-cyan
            "biobench-cyan"

        "DINOv2" ->
            -- bg-biobench-sea
            "biobench-sea"

        "AIMv2" ->
            -- bg-biobench-gold
            "biobench-gold"

        "CNN" ->
            -- bg-biobench-orange
            "biobench-orange"

        "cv4ecology" ->
            -- bg-biobench-cream
            "biobench-cream"

        "V-JEPA" ->
            -- bg-biobench-rust
            "biobench-rust"

        "SAM2" ->
            -- bg-biobench-scarlet
            "biobench-scarlet"

        _ ->
            -- bg-biobench-black
            "biobench-black"



-- CSS


toCssColorVar : String -> String
toCssColorVar color =
    "var(--color-" ++ color ++ ")"



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

            -- TODO:  release date, model family
            , { key = "params", display = "Params (M)", format = viewCheckpointParams, sortType = SortNumeric getCheckpointParams }
            , { key = "release", display = "Released", format = viewCheckpointRelease, sortType = SortNumeric getCheckpointRelease }
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
    D.map6 Checkpoint
        (D.field "ckpt" D.string)
        (D.field "display" D.string)
        (D.field "family" D.string)
        (D.succeed Nothing)
        (D.succeed Nothing)
        (D.succeed Nothing)


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



-- Components


viewFieldset : String -> List (Html Msg) -> Html Msg
viewFieldset title content =
    Html.fieldset
        [ class "border border-biobench-black p-2" ]
        [ Html.legend
            [ class "text-xs font-semibold tracking-tight px-1 -ml-1 " ]
            [ Html.text title ]
        , Html.div
            [ class "flex flex-wrap gap-x-4 gap-y-2" ]
            content
        ]


viewLabeledCheckbox : Bool -> (Bool -> Msg) -> String -> Html Msg
viewLabeledCheckbox checked msg label =
    Html.label
        [ class "inline-flex items-center gap-1 cursor-pointer select-none "
        ]
        [ viewCheckbox checked msg
        , Html.span
            [ class "text-sm tracking-tight" ]
            [ Html.text label ]
        ]


viewCheckbox : Bool -> (Bool -> Msg) -> Html Msg
viewCheckbox checked msg =
    Html.input
        [ Html.Attributes.type_ "checkbox"
        , Html.Attributes.checked checked
        , Html.Events.onCheck msg
        , class "accent-biobench-cyan cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-biobench-gold"
        ]
        []


viewLabeledRadio : String -> Bool -> (String -> Msg) -> String -> Html Msg
viewLabeledRadio value checked msg label =
    Html.label
        [ class "inline-flex items-center gap-1 cursor-pointer select-none "
        ]
        [ viewRadio value checked msg
        , Html.span
            [ class "text-sm tracking-tight" ]
            [ Html.text label ]
        ]


viewRadio : String -> Bool -> (String -> Msg) -> Html Msg
viewRadio value checked msg =
    Html.input
        [ Html.Attributes.type_ "radio"
        , Html.Attributes.checked checked
        , Html.Attributes.value value
        , Html.Events.onClick (msg value)
        , class "accent-biobench-cyan cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-biobench-gold"
        ]
        []
