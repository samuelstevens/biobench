module Leaderboard exposing (..)

import Browser
import Browser.Dom
import Browser.Events
import Chart as C
import Chart.Attributes as CA
import Dict
import Html exposing (Html)
import Html.Attributes as HA
import Html.Events as HE
import Html.Keyed
import Http
import Json.Decode as D
import Round
import Set
import Svg
import Svg.Attributes
import Task
import Time
import Trend.Linear
import Trend.Math


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }


type Msg
    = Fetched (Result Http.Error Table)
    | Sort String
    | ToggleFieldset Fieldset
    | ToggleCol String
    | ToggleFamily String
    | SetLayout Layout
    | MouseDown Float
    | DragStart DragInfo
    | DragMove Float
    | DragStop
    | NoOp


type Requested a e
    = Loading
    | Loaded a
    | Failed e


type alias Model =
    { requestedTable : Requested Table Http.Error

    -- Pickers
    , columnsFieldsetOpen : Bool
    , columnsSelected : Set.Set String
    , familiesFieldsetOpen : Bool
    , familiesSelected : Set.Set String
    , paramsOpen : Bool
    , paramCountRange : ( Int, Int )

    -- Sorting
    , sortKey : String
    , sortOrder : Order

    -- UI
    , layout : Layout
    , drag : Maybe DragInfo
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


type alias DragInfo =
    { x : Float
    , viewport : Browser.Dom.Viewport
    }


type Fieldset
    = ColumnsFieldset
    | FamiliesFieldset



-- | ParamsFieldset
-- | ReleaseFieldset
-- | ResolutionFieldset


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
    , resolution : Int
    , release : Maybe Time.Posix
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

    -- For graphing
    , barchart : Bool

    -- Quality of life
    , immediatelyVisible : Bool
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
      , columnsFieldsetOpen = False
      , columnsSelected = Set.empty

      -- TODO: implement
      , paramsOpen = True
      , paramCountRange = ( 0, 10 ^ 12 )
      , familiesFieldsetOpen = False
      , familiesSelected = Set.empty
      , sortKey = "mean"
      , sortOrder = Descending
      , layout = ChartsOnly
      , drag = Nothing
      }
    , Http.get
        { url = "data/results.json"
        , expect = Http.expectJson Fetched tableDecoder
        }
    )


subscriptions model =
    case model.drag of
        Nothing ->
            Sub.none

        Just _ ->
            Sub.batch
                [ Browser.Events.onMouseMove (D.map DragMove mouseX)
                , Browser.Events.onMouseUp (D.succeed DragStop)
                ]


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        Fetched result ->
            case result of
                Ok table ->
                    ( { model
                        | requestedTable = Loaded table
                        , columnsSelected =
                            table.cols
                                |> List.filter .immediatelyVisible
                                |> List.map .key
                                |> Set.fromList
                        , familiesSelected =
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

        ToggleFieldset fieldset ->
            case fieldset of
                ColumnsFieldset ->
                    ( { model | columnsFieldsetOpen = not model.columnsFieldsetOpen }, Cmd.none )

                FamiliesFieldset ->
                    ( { model | familiesFieldsetOpen = not model.familiesFieldsetOpen }, Cmd.none )

        ToggleCol key ->
            if Set.member key model.columnsSelected then
                ( { model | columnsSelected = Set.remove key model.columnsSelected }, Cmd.none )

            else
                ( { model | columnsSelected = Set.insert key model.columnsSelected }, Cmd.none )

        ToggleFamily key ->
            if Set.member key model.familiesSelected then
                ( { model | familiesSelected = Set.remove key model.familiesSelected }, Cmd.none )

            else
                ( { model | familiesSelected = Set.insert key model.familiesSelected }, Cmd.none )

        SetLayout layout ->
            ( { model | layout = layout }, Cmd.none )

        MouseDown x ->
            ( model
            , Task.attempt
                (\result ->
                    case result of
                        Err err ->
                            NoOp

                        Ok viewport ->
                            DragStart { x = x, viewport = viewport }
                )
                (Browser.Dom.getViewportOf "split-root")
            )

        DragStart info ->
            let
                pct =
                    (info.x - info.viewport.viewport.x) / info.viewport.viewport.width

                layout =
                    if pct < 0.02 then
                        ChartsOnly

                    else if pct > 0.98 then
                        TableOnly

                    else
                        Split pct
            in
            ( { model | drag = Just info, layout = layout }, Cmd.none )

        DragMove x ->
            case model.drag of
                Just info ->
                    let
                        pct =
                            (info.x - info.viewport.viewport.x) / info.viewport.viewport.width

                        layout =
                            if pct < 0.02 then
                                ChartsOnly

                            else if pct > 0.98 then
                                TableOnly

                            else
                                Split pct
                    in
                    ( { model | drag = Just { info | x = x }, layout = layout }, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        DragStop ->
            ( { model | drag = Nothing }, Cmd.none )


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
                    viewTable model.columnsSelected model.familiesSelected model.sortKey model.sortOrder table

                chartContent =
                    viewCharts model.columnsSelected model.familiesSelected table

                allFamilies =
                    table.rows
                        |> List.map (.checkpoint >> .family)
                        |> Set.fromList
                        |> Set.toList
                        |> List.sort
            in
            Html.div []
                [ Html.div
                    [ HA.class "flex flex-wrap gap-2" ]
                    [ viewWindowsFieldset model.layout
                    , viewCheckboxFieldset
                        ColumnsFieldset
                        model.columnsFieldsetOpen
                        model.columnsSelected
                        table.cols
                        (\col ->
                            viewCheckbox
                                (Set.member col.key model.columnsSelected)
                                (ToggleCol col.key)
                                col.display
                        )
                    , viewCheckboxFieldset FamiliesFieldset
                        model.familiesFieldsetOpen
                        model.familiesSelected
                        allFamilies
                        (\family ->
                            viewFamilyCheckbox
                                (Set.member family model.familiesSelected)
                                family
                        )
                    ]
                , Html.div
                    [ HA.class "flex", HA.id "split-root" ]
                    (case model.layout of
                        TableOnly ->
                            [ viewTablePane tableContent 100
                            , viewDragHandle model.drag
                            ]

                        ChartsOnly ->
                            [ viewDragHandle model.drag
                            , viewChartPane chartContent 100
                            ]

                        Split pct ->
                            [ viewTablePane tableContent ((pct - 0.01) * 100)
                            , viewDragHandle model.drag
                            , viewChartPane chartContent (100 - (pct + 0.01) * 100)
                            ]
                    )
                ]


viewTablePane : Html Msg -> Float -> Html Msg
viewTablePane content w =
    Html.div
        [ HA.style "width" (String.fromFloat w ++ "%"), HA.class "relative" ]
        [ Html.div [ HA.class "overflow-x-auto" ] [ content ]
        , -- Left fade
          Html.div [ HA.class "absolute left-0 top-0 bottom-0 w-5 z-1 pointer-events-none bg-gradient-to-r from-white to-transparent" ] []
        , -- Right fade
          Html.div [ HA.class "absolute right-0 top-0 bottom-0 w-5 z-1 pointer-events-none bg-gradient-to-l from-white to-transparent" ] []
        ]


viewChartPane : Html Msg -> Float -> Html Msg
viewChartPane content w =
    Html.div
        [ HA.style "width" (String.fromFloat w ++ "%")
        , HA.class "overflow-y-auto"
        ]
        [ content ]


viewDragHandle : Maybe DragInfo -> Html Msg
viewDragHandle info =
    let
        ( bg, fill ) =
            case info of
                Just _ ->
                    ( "bg-biobench-gold", "fill-biobench-gold" )

                Nothing ->
                    ( "bg-biobench-black", "fill-biobench-black" )
    in
    Html.div
        [ HA.class "hidden md:flex flex-col items-center group relative w-1 hover:bg-biobench-gold cursor-col-resize select-none z-40 drop-shadow-lg rounded-full"
        , HA.class bg
        , HE.on "mousedown" (D.map MouseDown mouseX)
        ]
        [ Html.div
            [ HA.class "mt-6" ]
            [ Svg.svg
                [ Svg.Attributes.viewBox "0 0 40 40 "
                , Svg.Attributes.width "40 "
                , Svg.Attributes.height "40 "
                ]
                [ Svg.circle
                    [ Svg.Attributes.cx "20 "
                    , Svg.Attributes.cy "20 "
                    , Svg.Attributes.r "20"
                    , Svg.Attributes.class "group-hover:fill-biobench-gold "
                    , Svg.Attributes.class fill
                    ]
                    []
                , Svg.polygon
                    [ Svg.Attributes.points "4,20 12,28 12,24 28,24 28,28 36,20 28,12, 28,16 12,16 12,12"
                    , Svg.Attributes.fill "white"
                    ]
                    []
                ]
            ]
        ]


mouseX : D.Decoder Float
mouseX =
    D.field "clientX" D.float


viewTable : Set.Set String -> Set.Set String -> String -> Order -> Table -> Html Msg
viewTable columnsSelected familiesSelected sortKey sortOrder table =
    Html.table
        [ HA.class "w-full md:text-sm mt-2" ]
        [ viewThead columnsSelected sortKey sortOrder table
        , viewTbody columnsSelected familiesSelected sortKey sortOrder table
        ]


viewThead : Set.Set String -> String -> Order -> Table -> Html Msg
viewThead columnsSelected sortKey sortOrder table =
    Html.thead
        [ HA.class "border-t border-b py-1 " ]
        (table.cols
            |> List.filter (\col -> Set.member col.key columnsSelected)
            |> List.map (viewTh sortKey sortOrder)
        )


viewTh : String -> Order -> TableCol -> Html Msg
viewTh sortKey sortOrder col =
    let
        ( suffix, extra ) =
            if sortKey == col.key then
                case sortOrder of
                    Descending ->
                        ( nonbreakingSpace ++ downArrow, "font-bold" )

                    Ascending ->
                        ( nonbreakingSpace ++ upArrow, "font-bold" )

            else
                ( "", "font-medium" )
    in
    Html.th
        [ HA.class "px-2", HA.class extra, HE.onClick (Sort col.key) ]
        [ Html.text (col.display ++ suffix) ]


viewTbody : Set.Set String -> Set.Set String -> String -> Order -> Table -> Html Msg
viewTbody columnsSelected familiesSelected sortKey sortOrder table =
    let
        filtered =
            table.rows
                |> List.filter (\row -> Set.member row.checkpoint.family familiesSelected)

        sortType =
            table.cols
                |> List.filter (\col -> col.key == sortKey)
                |> List.head
                |> Maybe.map .sortType
                |> Maybe.withDefault NotSortable

        sorted =
            case sortType of
                SortNumeric fn ->
                    List.sortBy (fn columnsSelected >> Maybe.withDefault (-1 / 0)) filtered

                SortString fn ->
                    List.sortBy (fn columnsSelected >> Maybe.withDefault maxString) filtered

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
        [ HA.class "border-b" ]
        (List.map
            (\row ->
                ( row.checkpoint.name
                , viewTr columnsSelected table.cols row
                )
            )
            ordered
        )


viewTr : Set.Set String -> List TableCol -> TableRow -> Html Msg
viewTr columnsSelected allCols row =
    let
        cols =
            allCols
                |> List.filter (\col -> Set.member col.key columnsSelected)
    in
    Html.tr
        [ HA.class "py-1 hover:bg-biobench-cream/20 transition-colors" ]
        (List.map (viewTd columnsSelected row) cols)


viewTd : Set.Set String -> TableRow -> TableCol -> Html Msg
viewTd columnsSelected row col =
    let
        ( cls, text ) =
            col.format columnsSelected row

        winner =
            if Set.member col.key row.winners then
                " font-bold"

            else
                ""
    in
    Html.td
        [ HA.class ("px-2 " ++ cls ++ winner) ]
        [ Html.text text ]


viewCharts : Set.Set String -> Set.Set String -> Table -> Html Msg
viewCharts columnsSelected familiesSelected table =
    let
        rows =
            table.rows
                |> List.filter (\r -> Set.member r.checkpoint.family familiesSelected)

        filteredCols =
            table.cols
                |> List.filter (\c -> Set.member c.key columnsSelected)

        ( getters, titles ) =
            filteredCols
                |> List.filter .barchart
                |> List.filterMap
                    (\c ->
                        case c.sortType of
                            SortNumeric fn ->
                                Just ( fn columnsSelected, c.display )

                            _ ->
                                Nothing
                    )
                |> List.unzip

        barCharts =
            List.map2 (viewBarChart rows) titles getters

        functions =
            filteredCols
                |> List.filterMap
                    (\c ->
                        case c.sortType of
                            SortNumeric fn ->
                                Just ( c.key, fn columnsSelected )

                            _ ->
                                Nothing
                    )
                |> Dict.fromList

        scatterData : List (Dict.Dict String Float)
        scatterData =
            List.map
                (\row ->
                    Dict.foldl
                        (\key func acc ->
                            case func row of
                                Just value ->
                                    Dict.insert key value acc

                                Nothing ->
                                    acc
                        )
                        Dict.empty
                        functions
                )
                rows

        scatterCharts =
            [ viewScatterChart "imagenet1k" "newt" scatterData
            , viewScatterChart "imagenet1k" "mean" scatterData
            ]
    in
    Html.div
        [ HA.class "grid gap-2 [grid-template-columns:repeat(auto-fit,minmax(16rem,1fr))] mt-t" ]
        -- (scatterCharts ++ barCharts)
        barCharts


viewBarChart : List TableRow -> String -> (TableRow -> Maybe Float) -> Html Msg
viewBarChart rows title getter =
    case List.filterMap (withInfo getter) rows of
        [] ->
            Html.div [ HA.class "hidden" ] []

        data ->
            Html.div [ HA.class "" ]
                [ C.chart
                    [ CA.height 300
                    , CA.width 300
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
                        [ HA.class "" ]
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
                        [ CA.fontSize 24, CA.moveUp 2 ]
                        [ Svg.text title ]
                    ]
                ]


lineToPoints : Trend.Linear.Line -> List ( Float, Float )
lineToPoints line =
    [ ( 0, Trend.Linear.predictY line 0 )
    , ( 100, Trend.Linear.predictY line 100 )
    ]


viewScatterChart : String -> String -> List (Dict.Dict String Float) -> Html Msg
viewScatterChart x y raw =
    let
        points =
            raw
                |> List.filterMap
                    (\d ->
                        case ( getScatterField x d, getScatterField y d ) of
                            ( Just x_, Just y_ ) ->
                                Just ( x_, y_ )

                            ( _, _ ) ->
                                Nothing
                    )

        fullTrendPoints : List ( Float, Float )
        fullTrendPoints =
            points
                |> Trend.Linear.quick
                |> Result.map Trend.Linear.line
                |> Result.map lineToPoints
                |> Result.withDefault []

        partTrendPoints : List ( Float, Float )
        partTrendPoints =
            points
                |> List.filter (Tuple.first >> (<) 75)
                |> Trend.Linear.quick
                |> Result.map Trend.Linear.line
                |> Result.map lineToPoints
                |> Result.withDefault []
    in
    C.chart
        [ CA.height 300
        , CA.width 300
        , CA.margin
            { top = 25
            , bottom = 10
            , left = 37
            , right = 10
            }
        , CA.range
            [ CA.lowest 0 CA.exactly
            , CA.highest 100 CA.exactly
            ]
        , CA.domain
            [ CA.lowest 0 CA.exactly
            , CA.highest 100 CA.exactly
            ]
        ]
        [ C.xLabels [ CA.withGrid, CA.amount 3 ]
        , C.yLabels [ CA.withGrid, CA.amount 3 ]
        , C.series Tuple.first
            [ C.scatter Tuple.second [] ]
            points
        , C.series Tuple.first
            [ C.interpolated Tuple.second [] [] ]
            fullTrendPoints
        , C.series Tuple.first
            [ C.interpolated Tuple.second [] [] ]
            partTrendPoints
        ]


getScatterField : String -> Dict.Dict String Float -> Maybe Float
getScatterField field data =
    Dict.get field data
        |> Maybe.map ((*) 100.0)


withInfo : (TableRow -> Maybe Float) -> TableRow -> Maybe { score : Float, family : String, display : String }
withInfo getter row =
    case getter row of
        Nothing ->
            Nothing

        Just score ->
            Just
                { score = score
                , family = row.checkpoint.family
                , display = row.checkpoint.display
                }


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
getMeanScore tasks columnsSelected row =
    let
        selectedScores =
            tasks
                |> Set.intersect columnsSelected
                |> Set.toList
                |> List.filterMap (\key -> Dict.get key row.scores)
    in
    -- columnsSelected can be more than 9 (includes imagenet1k, newt, checkpoint, mean, etc).
    -- I think I might have to hardcode the benchmark tasks somewhere.
    if List.length selectedScores == (tasks |> Set.intersect columnsSelected |> Set.size) then
        mean selectedScores

    else
        Nothing


viewMeanScore : Set.Set String -> Set.Set String -> TableRow -> ( String, String )
viewMeanScore tasks columnsSelected row =
    ( "text-right font-mono"
    , getMeanScore tasks columnsSelected row |> viewScore
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
    row.checkpoint.params |> toFloat |> Just


viewCheckpointParams : Set.Set String -> TableRow -> ( String, String )
viewCheckpointParams columnsSelected row =
    ( "text-right"
    , toFloat row.checkpoint.params
        / (10 ^ 6)
        |> Round.round 0
    )


getCheckpointRelease : Set.Set String -> TableRow -> Maybe Float
getCheckpointRelease _ row =
    row.checkpoint.release |> Maybe.map (Time.posixToMillis >> toFloat)


viewCheckpointRelease : Set.Set String -> TableRow -> ( String, String )
viewCheckpointRelease _ row =
    ( "text-right"
    , row.checkpoint.release
        |> Maybe.map (formatTime Time.utc)
        |> Maybe.withDefault "unknown"
    )


getCheckpointResolution : Set.Set String -> TableRow -> Maybe Float
getCheckpointResolution _ row =
    row.checkpoint.resolution |> toFloat |> Just


viewCheckpointResolution : Set.Set String -> TableRow -> ( String, String )
viewCheckpointResolution _ row =
    ( "text-right"
    , String.fromInt row.checkpoint.resolution
    )


formatTime : Time.Zone -> Time.Posix -> String
formatTime zone posix =
    (Time.toMonth zone posix |> formatMonth) ++ nonbreakingSpace ++ (Time.toYear zone posix |> String.fromInt)


formatMonth : Time.Month -> String
formatMonth month =
    case month of
        Time.Jan ->
            "Jan"

        Time.Feb ->
            "Feb"

        Time.Mar ->
            "March"

        Time.Apr ->
            "April"

        Time.May ->
            "May"

        Time.Jun ->
            "June"

        Time.Jul ->
            "July"

        Time.Aug ->
            "Aug"

        Time.Sep ->
            "Sep"

        Time.Oct ->
            "Oct"

        Time.Nov ->
            "Nov"

        Time.Dec ->
            "Dec"


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


rightArrow =
    "▶"


maxString : String
maxString =
    String.repeat 50 "\u{10FFFF}"


nonbreakingSpace : String
nonbreakingSpace =
    "\u{00A0}"


nonbreakingDash : String
nonbreakingDash =
    "‑"


familyColor : String -> String
familyColor fam =
    -- Include all bg-VAR as comments to force tailwind to include all the colors as variables in the final CSS.
    -- This is an example of a comment changing system behavior!!
    case fam of
        "CLIP" ->
            -- accent-biobench-blue
            -- focus-visible:outline-biobench-blue
            "biobench-blue"

        "SigLIP" ->
            -- accent-biobench-cyan
            -- focus-visible:outline-biobench-cyan
            "biobench-cyan"

        "DINOv2" ->
            -- accent-biobench-sea
            -- focus-visible:outline-biobench-sea
            "biobench-sea"

        "AIMv2" ->
            -- accent-biobench-gold
            -- focus-visible:outline-biobench-gold
            "biobench-gold"

        "CNN" ->
            -- accent-biobench-orange
            -- focus-visible:outline-biobench-orange
            "biobench-orange"

        "cv4ecology" ->
            -- accent-biobench-cream
            -- focus-visible:outline-biobench-cream
            "biobench-cream"

        "V-JEPA" ->
            -- accent-biobench-rust
            -- focus-visible:outline-biobench-rust
            "biobench-rust"

        "SAM2" ->
            -- accent-biobench-scarlet
            -- focus-visible:outline-biobench-scarlet
            "biobench-scarlet"

        _ ->
            -- accent-biobench-black
            -- focus-visible:outline-biobench-black
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
                    , barchart = True
                    , immediatelyVisible = True
                    }
                )
                payload.benchmarkTasks

        tasks =
            payload.benchmarkTasks |> List.map .name |> Set.fromList

        fixedCols =
            [ { key = "checkpoint", display = "Checkpoint", format = viewCheckpoint, sortType = SortString getCheckpoint, barchart = False, immediatelyVisible = True }
            , { key = "params", display = "Params" ++ nonbreakingSpace ++ "(M)", format = viewCheckpointParams, sortType = SortNumeric getCheckpointParams, barchart = False, immediatelyVisible = False }
            , { key = "resolution", display = "Res." ++ nonbreakingSpace ++ "(px)", format = viewCheckpointResolution, sortType = SortNumeric getCheckpointResolution, barchart = False, immediatelyVisible = False }
            , { key = "release", display = "Released", format = viewCheckpointRelease, sortType = SortNumeric getCheckpointRelease, barchart = False, immediatelyVisible = False }
            , { key = "imagenet1k", display = "ImageNet" ++ nonbreakingDash ++ "1K", format = viewBenchmarkScore "imagenet1k", sortType = SortNumeric (getBenchmarkScore "imagenet1k"), barchart = True, immediatelyVisible = False }
            , { key = "newt", display = "NeWT", format = viewBenchmarkScore "newt", sortType = SortNumeric (getBenchmarkScore "newt"), barchart = True, immediatelyVisible = False }
            , { key = "mean", display = "Mean", format = viewMeanScore tasks, sortType = SortNumeric (getMeanScore tasks), barchart = True, immediatelyVisible = True }
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
        (D.map
            (String.replace " " nonbreakingSpace
                >> String.replace "-" nonbreakingDash
            )
            (D.field "display" D.string)
        )
        (D.field "family" D.string)
        (D.field "params" D.int)
        (D.field "resolution" D.int)
        (D.field "release_ms"
            (D.maybe (D.map Time.millisToPosix D.int))
        )


timeDecoder : D.Decoder Time.Posix
timeDecoder =
    D.map Time.millisToPosix D.int


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


viewCheckboxFieldset : Fieldset -> Bool -> Set.Set String -> List a -> (a -> Html Msg) -> Html Msg
viewCheckboxFieldset fieldset open checked checkables checkboxOf =
    let
        clickHandler =
            HE.onClick (ToggleFieldset fieldset)

        arrow =
            if open then
                downArrow

            else
                rightArrow

        commonButtonAttrs =
            [ HA.class "flex item-centers gap-1 font-semibold px-2 py-1 text-sm cursor-pointer rounded-sm leading-none hover:bg-gray-50 group-hover:bg-gray-50 transition-colors duration-100"
            ]

        buttonAttrs =
            if open then
                commonButtonAttrs ++ [ clickHandler ]

            else
                commonButtonAttrs

        summary =
            "(" ++ (checked |> Set.size |> String.fromInt) ++ "/" ++ (checkables |> List.length |> String.fromInt) ++ ")"

        legend =
            Html.legend []
                [ Html.button
                    buttonAttrs
                    [ Html.text (viewFieldset fieldset)
                    , Html.span
                        [ HA.class "text-black text-sm leading-none select-none"
                        , HA.attribute "aria-hidden" "true"
                        ]
                        [ Html.text arrow ]
                    , if not open then
                        Html.span
                            [ HA.class "text-sm text-gray-600 font-normal leading-none" ]
                            [ Html.text summary ]

                      else
                        Html.text ""
                    ]
                ]

        openContent =
            Html.div
                [ HA.class "flex flex-wrap gap-x-2 md:gap-x-4 gap-y-2" ]
                (List.map checkboxOf checkables)

        closedContent =
            Html.div
                [ HA.class "text-center text-sm text-gray-600 cursor-pointer hover:bg-gray-50 transition-colors duration-100" ]
                [ Html.text "Click to expand" ]

        commonFieldsetClass =
            "border border-black p-2 bg-white transition-colors duration-150 "

        fieldsetClass =
            if open then
                commonFieldsetClass

            else
                commonFieldsetClass ++ "group cursor-pointer hover:bg-gray-50"
    in
    if open then
        Html.fieldset
            [ HA.class fieldsetClass ]
            [ legend, openContent ]

    else
        Html.fieldset
            [ HA.class fieldsetClass, clickHandler ]
            [ legend, closedContent ]


viewCheckbox : Bool -> Msg -> String -> Html Msg
viewCheckbox checked msg label =
    Html.label
        [ HA.class "inline-flex items-center sm:gap-1 cursor-pointer select-none "
        , HE.custom "click" (D.succeed { message = msg, stopPropagation = True, preventDefault = True })
        ]
        [ Html.input
            [ HA.type_ "checkbox"
            , HA.checked checked
            , HA.class checkboxClass
            ]
            []
        , Html.span [ HA.class "md:text-sm " ] [ Html.text label ]
        ]


viewFamilyCheckbox : Bool -> String -> Html Msg
viewFamilyCheckbox checked family =
    Html.label
        [ HA.class "inline-flex items-center sm:gap-1 cursor-pointer select-none "
        , HE.custom "click" (D.succeed { message = ToggleFamily family, stopPropagation = True, preventDefault = True })
        ]
        [ Html.input
            [ HA.type_ "checkbox"
            , HA.checked checked
            , HA.class "cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 "
            , HA.class ("accent-" ++ familyColor family)
            , HA.class ("focus-visible:outline-" ++ familyColor family)
            ]
            []
        , Html.span [ HA.class "md:text-sm " ] [ Html.text family ]
        ]


viewWindowsFieldset : Layout -> Html Msg
viewWindowsFieldset layout =
    let
        legend =
            Html.legend []
                [ Html.span
                    [ HA.class "flex item-centers gap-1 font-semibold px-2 py-1 text-sm cursor-pointer rounded-sm leading-none"
                    ]
                    [ Html.text "Windows" ]
                ]
    in
    Html.fieldset
        [ HA.class "border border-black p-2 bg-white transition-colors duration-150 " ]
        [ legend
        , Html.div
            [ HA.class "flex flex-wrap gap-x-2 md:gap-x-4 gap-y-2" ]
            [ viewLabeledRadio
                "table-only"
                (layoutEq layout TableOnly)
                (\_ -> SetLayout TableOnly)
                "Tables"
            , Html.label
                [ HA.class "hidden md:inline-flex items-center gap-1 cursor-pointer select-none "
                ]
                [ Html.input
                    [ HA.type_ "radio"
                    , HA.checked (layoutEq layout (Split -1))
                    , HA.value "split"
                    , HE.onClick (SetLayout (Split 0.5))
                    , HA.class radioClass
                    ]
                    []
                , Html.span
                    [ HA.class "text-sm " ]
                    [ Html.text "Split" ]
                ]
            , viewLabeledRadio
                "charts-only"
                (layoutEq layout ChartsOnly)
                (\_ -> SetLayout ChartsOnly)
                "Charts"
            ]
        ]


viewFieldset : Fieldset -> String
viewFieldset fieldset =
    case fieldset of
        ColumnsFieldset ->
            "Columns"

        FamiliesFieldset ->
            "Model Families"


viewLabeledRadio : String -> Bool -> (String -> Msg) -> String -> Html Msg
viewLabeledRadio value checked msg label =
    Html.label
        [ HA.class "inline-flex items-center sm:gap-1 cursor-pointer select-none " ]
        [ Html.input
            [ HA.type_ "radio"
            , HA.checked checked
            , HA.value value
            , HE.onClick (msg value)
            , HA.class radioClass
            ]
            []
        , Html.span [ HA.class "md:text-sm " ] [ Html.text label ]
        ]



-- CSS


radioClass =
    "accent-biobench-cyan cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-biobench-gold"


checkboxClass =
    "accent-biobench-cyan cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-biobench-cyan"
