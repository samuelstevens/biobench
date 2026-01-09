module Leaderboard exposing (..)

import Browser
import Browser.Dom
import Browser.Events
import Chart as C
import Chart.Attributes as CA
import Chart.Events as CE
import Chart.Item as CI
import Dict exposing (Dict)
import Html exposing (Html)
import Html.Attributes as HA
import Html.Events as HE
import Html.Keyed
import Http
import Json.Decode as D
import Round
import Set exposing (Set)
import Svg
import Svg.Attributes as SA
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
    = NoOp
    | Fetched (Result Http.Error Table)
    | Sort String
    | ToggleFieldset Fieldset
    | ToggleCol String
    | ToggleFamily String
    | ToggleParamRange ( Int, Int )
    | ToggleResolution Int
    | SetLayout Layout
    | MouseDown Float
    | DragStart DragInfo
    | DragMove Float
    | DragStop
    | OnHoverBar (Maybe String) (List (CI.One BarDatum CI.Bar))
    | OnHoverDot (Maybe String) (List (CI.One DotDatum CI.Dot))


type Requested a e
    = Loading
    | Loaded a
    | Failed e


type alias Model =
    { requestedTable : Requested Table Http.Error

    -- Pickers
    -- Columns
    , columnsFieldsetOpen : Bool
    , columnsSelected : Set String

    -- Model familes
    , familiesFieldsetOpen : Bool
    , familiesSelected : Set String

    -- Param ranges
    , paramRangesFieldsetOpen : Bool
    , paramRangesSelected : Set ( Int, Int )

    -- Resolutions
    , resolutionsFieldsetOpen : Bool
    , resolutionsSelected : Set Int

    -- Sorting
    , sortKey : String
    , sortOrder : Order

    -- UI
    , layout : Layout
    , drag : Maybe DragInfo

    -- Charts
    , hoveredKey : Maybe String
    , hoveredBars : List (CI.One BarDatum CI.Bar)
    , hoveredDots : List (CI.One DotDatum CI.Dot)
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
      -- | ReleaseFieldset
    | ParamRangesFieldset
    | ResolutionsFieldset


type alias Table =
    { rows : List TableRow
    , cols : List TableCol
    , metadata : Metadata
    }


type alias TableRow =
    { checkpoint : Checkpoint
    , scores : Dict String Float
    , winners : Set String
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
    = SortNumeric (Set String -> TableRow -> Maybe Float)
    | SortString (Set String -> TableRow -> Maybe String)
    | NotSortable


type alias TableCol =
    { key : String
    , display : String

    -- How to get the cell value
    -- (results in a class string and an Html.text string)
    , format : Set String -> TableRow -> ( String, String )

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
      , columnsFieldsetOpen = True
      , columnsSelected = Set.empty
      , paramRangesFieldsetOpen = False
      , paramRangesSelected = Set.fromList allParamRanges
      , familiesFieldsetOpen = True
      , familiesSelected = Set.empty
      , resolutionsFieldsetOpen = False
      , resolutionsSelected = Set.empty
      , sortKey = "mean"
      , sortOrder = Descending
      , layout = TableOnly
      , drag = Nothing
      , hoveredKey = Nothing
      , hoveredBars = []
      , hoveredDots = []
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
                        , resolutionsSelected =
                            table.rows
                                |> List.map (.checkpoint >> .resolution)
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
                ( { model | sortKey = key, sortOrder = Descending }, Cmd.none )

        ToggleFieldset fieldset ->
            case fieldset of
                ColumnsFieldset ->
                    ( { model | columnsFieldsetOpen = not model.columnsFieldsetOpen }, Cmd.none )

                FamiliesFieldset ->
                    ( { model | familiesFieldsetOpen = not model.familiesFieldsetOpen }, Cmd.none )

                ParamRangesFieldset ->
                    ( { model | paramRangesFieldsetOpen = not model.paramRangesFieldsetOpen }, Cmd.none )

                ResolutionsFieldset ->
                    ( { model | resolutionsFieldsetOpen = not model.resolutionsFieldsetOpen }, Cmd.none )

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

        ToggleParamRange range ->
            if Set.member range model.paramRangesSelected then
                ( { model | paramRangesSelected = Set.remove range model.paramRangesSelected }, Cmd.none )

            else
                ( { model | paramRangesSelected = Set.insert range model.paramRangesSelected }, Cmd.none )

        ToggleResolution range ->
            if Set.member range model.resolutionsSelected then
                ( { model | resolutionsSelected = Set.remove range model.resolutionsSelected }, Cmd.none )

            else
                ( { model | resolutionsSelected = Set.insert range model.resolutionsSelected }, Cmd.none )

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

        OnHoverBar key hoveredBars ->
            ( { model | hoveredBars = hoveredBars, hoveredKey = key }, Cmd.none )

        OnHoverDot key hoveredDots ->
            ( { model | hoveredDots = hoveredDots, hoveredKey = key }, Cmd.none )


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
                    viewTable
                        model.columnsSelected
                        model.familiesSelected
                        model.paramRangesSelected
                        model.resolutionsSelected
                        model.sortKey
                        model.sortOrder
                        table

                chartContent =
                    viewCharts model.hoveredKey
                        model.hoveredBars
                        model.hoveredDots
                        model.columnsSelected
                        model.familiesSelected
                        model.paramRangesSelected
                        model.resolutionsSelected
                        table

                allFamilies =
                    table.rows
                        |> List.map (.checkpoint >> .family)
                        |> Set.fromList
                        |> Set.toList
                        |> List.sort

                allResolutions =
                    table.rows
                        |> List.map (.checkpoint >> .resolution)
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
                    , viewCheckboxFieldset ParamRangesFieldset
                        model.paramRangesFieldsetOpen
                        model.paramRangesSelected
                        allParamRanges
                        (\range ->
                            viewCheckbox
                                (Set.member range model.paramRangesSelected)
                                (ToggleParamRange range)
                                (formatRange range)
                        )
                    , viewCheckboxFieldset ResolutionsFieldset
                        model.resolutionsFieldsetOpen
                        model.resolutionsSelected
                        allResolutions
                        (\res ->
                            viewCheckbox
                                (Set.member res model.resolutionsSelected)
                                (ToggleResolution res)
                                (String.fromInt res ++ "px")
                        )
                    ]
                , Html.div
                    [ HA.class "flex mt-2", HA.id "split-root" ]
                    (case model.layout of
                        TableOnly ->
                            [ viewTablePane tableContent 100
                            , viewDragHandle model.drag
                            ]

                        ChartsOnly ->
                            [ viewDragHandle model.drag
                            , viewChartsPane chartContent 100
                            ]

                        Split pct ->
                            [ viewTablePane tableContent ((pct - 0.01) * 100)
                            , viewDragHandle model.drag
                            , viewChartsPane chartContent (100 - (pct + 0.01) * 100)
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


viewChartsPane : Html Msg -> Float -> Html Msg
viewChartsPane content w =
    Html.div
        [ HA.style "width" (String.fromFloat w ++ "%")
        , HA.class "overflow-y-auto pl-2"
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
                    ( "bg-gray-500", "fill-gray-500" )
    in
    Html.div
        [ HA.class "hidden md:flex flex-col items-center group relative w-1 hover:bg-biobench-gold cursor-col-resize select-none z-40 drop-shadow-lg rounded-full"
        , HA.class bg
        , HE.on "mousedown" (D.map MouseDown mouseX)
        ]
        [ Html.div
            [ HA.class "mt-6" ]
            [ Svg.svg
                [ SA.viewBox "0 0 40 40 "
                , SA.width "40 "
                , SA.height "40 "
                ]
                [ Svg.circle
                    [ SA.cx "20 "
                    , SA.cy "20 "
                    , SA.r "20"
                    , SA.class "group-hover:fill-biobench-gold "
                    , SA.class fill
                    ]
                    []
                , Svg.polygon
                    [ SA.points "4,20 12,28 12,24 28,24 28,28 36,20 28,12, 28,16 12,16 12,12"
                    , SA.fill "white"
                    ]
                    []
                ]
            ]
        ]


mouseX : D.Decoder Float
mouseX =
    D.field "clientX" D.float


viewTable : Set String -> Set String -> Set ( Int, Int ) -> Set Int -> String -> Order -> Table -> Html Msg
viewTable columnsSelected familiesSelected paramRangesSelected resolutionsSelected sortKey sortOrder table =
    Html.table
        [ HA.class "w-full md:text-sm " ]
        [ viewThead columnsSelected sortKey sortOrder table
        , viewTbody
            columnsSelected
            familiesSelected
            paramRangesSelected
            resolutionsSelected
            sortKey
            sortOrder
            table
        ]


viewThead : Set String -> String -> Order -> Table -> Html Msg
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


viewTbody : Set String -> Set String -> Set ( Int, Int ) -> Set Int -> String -> Order -> Table -> Html Msg
viewTbody columnsSelected familiesSelected paramRangesSelected resolutionsSelected sortKey sortOrder table =
    let
        filtered =
            table.rows
                |> List.filter (\row -> Set.member row.checkpoint.family familiesSelected)
                |> List.filter (.checkpoint >> .params >> paramRangesMatch paramRangesSelected)
                |> List.filter (\row -> Set.member row.checkpoint.resolution resolutionsSelected)

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
                ( row.checkpoint.name, viewTr columnsSelected table.cols row )
            )
            ordered
        )


viewTr : Set String -> List TableCol -> TableRow -> Html Msg
viewTr columnsSelected allCols row =
    let
        cols =
            allCols
                |> List.filter (\col -> Set.member col.key columnsSelected)
    in
    Html.tr
        [ HA.class "py-1 hover:bg-biobench-cream/20 transition-colors" ]
        (List.map (viewTd columnsSelected row) cols)


viewTd : Set String -> TableRow -> TableCol -> Html Msg
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


viewCharts : Maybe String -> List (CI.One BarDatum CI.Bar) -> List (CI.One DotDatum CI.Dot) -> Set String -> Set String -> Set ( Int, Int ) -> Set Int -> Table -> Html Msg
viewCharts hoveredKey hoveredBars hoveredDots columnsSelected familiesSelected paramRangesSelected resolutionsSelected table =
    let
        rows =
            table.rows
                |> List.filter (\r -> Set.member r.checkpoint.family familiesSelected)
                |> List.filter (.checkpoint >> .params >> paramRangesMatch paramRangesSelected)
                |> List.filter (\row -> Set.member row.checkpoint.resolution resolutionsSelected)

        filteredCols : List TableCol
        filteredCols =
            table.cols
                |> List.filter (\c -> Set.member c.key columnsSelected)

        barCharts : List (Html Msg)
        barCharts =
            filteredCols
                |> List.filter .barchart
                |> List.filterMap (viewMaybeBarChart hoveredKey hoveredBars columnsSelected rows)

        scatterData : List ( PointMetadata, Dict String Float )
        scatterData =
            List.map
                (\row ->
                    ( rowToPointMetadata row
                    , List.foldl
                        (\col acc ->
                            case col.sortType of
                                SortNumeric fn ->
                                    case fn columnsSelected row of
                                        Just value ->
                                            Dict.insert col.key value acc

                                        Nothing ->
                                            acc

                                _ ->
                                    acc
                        )
                        Dict.empty
                        filteredCols
                    )
                )
                rows

        scatterCharts =
            [ viewImagenetCorrelationChart hoveredKey hoveredDots ( "newt", "NeWT" ) scatterData
            , viewImagenetCorrelationChart hoveredKey hoveredDots ( "inat21", "iNat21" ) scatterData
            , viewImagenetCorrelationChart hoveredKey hoveredDots ( "mean", "Mean" ) scatterData
            , viewParamsCorrelationChart hoveredKey hoveredDots ( "imagenet1k", "ImageNet-1K" ) scatterData
            , viewParamsCorrelationChart hoveredKey hoveredDots ( "newt", "NeWT" ) scatterData
            , viewParamsCorrelationChart hoveredKey hoveredDots ( "inat21", "iNat21" ) scatterData
            , viewParamsCorrelationChart hoveredKey hoveredDots ( "mean", "Mean" ) scatterData
            ]
    in
    Html.div
        [ HA.class "grid gap-2 [grid-template-columns:repeat(auto-fit,minmax(20rem,1fr))] " ]
        (scatterCharts ++ barCharts)


viewMaybeBarChart : Maybe String -> List (CI.One BarDatum CI.Bar) -> Set String -> List TableRow -> TableCol -> Maybe (Html Msg)
viewMaybeBarChart hoveredKey hoveredBars columnsSelected rows col =
    case col.sortType of
        SortNumeric fn ->
            case List.filterMap (toBarDatum (fn columnsSelected)) rows of
                [] ->
                    Nothing

                data ->
                    Just
                        (viewBarChart
                            (filterHovered col.display hoveredKey hoveredBars)
                            col.display
                            data
                        )

        _ ->
            Nothing


viewBarChart : List (CI.One BarDatum CI.Bar) -> String -> List BarDatum -> Html Msg
viewBarChart hovered title data =
    Html.div [ HA.class "" ]
        [ C.chart
            [ CA.height 300
            , CA.width 300
            , CA.margin
                { top = 25
                , bottom = 10
                , left = 45
                , right = 10
                }
            , CA.domain
                [ CA.lowest 0 CA.exactly
                , CA.highest 100 CA.exactly
                ]
            , CE.onMouseMove (OnHoverBar (Just title)) (CE.getNearestX CI.bars)
            , CE.onMouseLeave (OnHoverBar Nothing [])
            ]
            [ C.yLabels [ CA.amount 3 ]
            , C.yTicks [ CA.amount 3 ]
            , C.bars
                []
                [ C.bar (Tuple.second >> .score >> (*) 100)
                    [ CA.opacity 0.8 ]
                    |> C.variation
                        (\index ( metadata, datum ) ->
                            [ CA.color (metadata.family |> familyColor |> toCssColorVar) ]
                        )
                    |> C.amongst hovered
                        (\_ -> [ CA.opacity 1.0 ])
                ]
                data
            , C.each hovered
                (\p bar ->
                    let
                        color =
                            CI.getColor bar

                        name =
                            CI.getData bar |> Tuple.first |> .display

                        score =
                            CI.getY bar
                    in
                    [ C.tooltip bar
                        [ CA.onTop, CA.top, CA.offset 0 ]
                        [ HA.style "color" color ]
                        [ Html.text name
                        , Html.text ": "
                        , Html.text (Round.round 1 score)
                        ]
                    ]
                )
            , C.labelAt
                .min
                CA.middle
                [ CA.moveLeft 35, CA.rotate 90 ]
                [ Svg.text title ]
            ]
        ]


lineToPoints : ( Float, Float ) -> Trend.Linear.Line -> List { x : Float, y : Float }
lineToPoints ( low, high ) line =
    [ { x = low, y = Trend.Linear.predictY line low }
    , { x = high, y = Trend.Linear.predictY line high }
    ]


viewImagenetCorrelationChart : Maybe String -> List (CI.One DotDatum CI.Dot) -> ( String, String ) -> List ( PointMetadata, Dict String Float ) -> Html Msg
viewImagenetCorrelationChart hoveredKey hoveredDots ( field, title ) data =
    let
        key =
            "imagenet1k-vs-" ++ field

        hovered =
            filterHovered key hoveredKey hoveredDots

        pointsName =
            "points"
    in
    case List.filterMap (toDot "imagenet1k" ((*) 100) field ((*) 100)) data of
        [] ->
            Html.div [ HA.class "hidden" ] []

        points ->
            let
                trend : Result Trend.Math.Error (Trend.Linear.Trend Trend.Linear.Quick)
                trend =
                    points
                        |> List.map (\( meta, p ) -> ( p.x, p.y ))
                        |> Trend.Linear.quick

                trendPoints : List DotDatum
                trendPoints =
                    trend
                        |> Result.map Trend.Linear.line
                        |> Result.map (lineToPoints ( 0, 100 ))
                        |> Result.withDefault []
                        |> List.map (\p -> ( { family = "", display = "" }, p ))

                ( rSqAttrs, rSqHtml ) =
                    case Result.map Trend.Linear.goodnessOfFit trend of
                        Err err ->
                            ( [ HA.class "font-mono text-red-500" ]
                            , [ Html.text "Error: "
                              , Html.text <| formatTrendErr err
                              ]
                            )

                        Ok rSq ->
                            ( [ HA.class "font-mono text-blue-400 text-sm" ]
                            , [ Html.text "R^2="
                              , Html.text <| Round.round 2 rSq
                              ]
                            )

                rSqLabel =
                    C.htmlAt .max .min -75 20 rSqAttrs rSqHtml
            in
            C.chart
                [ CA.height 300
                , CA.width 300
                , CA.margin
                    { top = 25
                    , bottom = 30
                    , left = 45
                    , right = 10
                    }

                -- , CA.range
                --     [ CA.lowest 0 CA.exactly
                --     , CA.highest 100 CA.exactly
                --     ]
                -- , CA.domain
                --     [ CA.lowest 0 CA.exactly
                --     , CA.highest 100 CA.exactly
                --     ]
                , CI.dots
                    |> CI.andThen (CI.named [ pointsName ])
                    |> CE.getNearest
                    |> CE.onMouseMove (OnHoverDot (Just key))
                , CE.onMouseLeave (OnHoverDot Nothing [])
                ]
                [ C.xLabels [ CA.withGrid, CA.amount 3 ]
                , C.yLabels [ CA.withGrid, CA.amount 3 ]
                , C.series (Tuple.second >> .x)
                    [ C.interpolated (Tuple.second >> .y)
                        -- bg-blue-300
                        [ CA.color "var(--color-blue-300)", CA.width 2 ]
                        []
                        |> C.named "trendline"
                    ]
                    trendPoints
                , C.series (Tuple.second >> .x)
                    [ C.scatter (Tuple.second >> .y)
                        [ CA.opacity 0.5, CA.borderWidth 1 ]
                        |> C.named pointsName
                        |> C.variation
                            (\index ( metadata, datum ) ->
                                let
                                    color =
                                        metadata.family |> familyColor |> toCssColorVar
                                in
                                [ CA.color color, CA.border color ]
                            )
                        |> C.amongst hovered
                            (\_ -> [ CA.highlight 0.4, CA.opacity 1.0 ])
                    ]
                    points
                , C.each hovered
                    (\p dot ->
                        let
                            name =
                                CI.getData dot |> Tuple.first |> .display
                        in
                        [ C.tooltip dot
                            [ CA.offset 0 ]
                            [ HA.style "color" <| CI.getColor dot ]
                            [ Html.text name
                            , Html.text " ("
                            , Html.text (Round.round 1 <| CI.getX dot)
                            , Html.text ", "
                            , Html.text (Round.round 1 <| CI.getY dot)
                            , Html.text ")"
                            ]
                        ]
                    )
                , rSqLabel

                -- , C.series Tuple.first
                --     [ C.interpolated Tuple.second [] [] ]
                --     partTrendPoints
                , C.labelAt
                    CA.middle
                    .min
                    [ CA.moveDown 35 ]
                    [ Svg.text "ImageNet-1K" ]
                , C.labelAt
                    .min
                    CA.middle
                    [ CA.moveLeft 35, CA.rotate 90 ]
                    [ Svg.text title ]
                ]


viewParamsCorrelationChart : Maybe String -> List (CI.One DotDatum CI.Dot) -> ( String, String ) -> List ( PointMetadata, Dict String Float ) -> Html Msg
viewParamsCorrelationChart hoveredKey hoveredDots ( field, title ) data =
    let
        -- fullTrendPoints : List ( Float, Float )
        -- fullTrendPoints =
        --     points
        --         |> List.map (\p -> ( p.x, p.y ))
        --         |> Trend.Linear.quick
        --         |> Result.map Trend.Linear.line
        --         |> Result.map (lineToPoints 0)
        --         |> Result.withDefault []
        key =
            "params-vs-" ++ field

        hovered =
            filterHovered key hoveredKey hoveredDots
    in
    case List.filterMap (toDot "params" (logBase 10) field ((*) 100)) data of
        [] ->
            Html.div [ HA.class "hidden" ] []

        points ->
            let
                trend : Result Trend.Math.Error (Trend.Linear.Trend Trend.Linear.Quick)
                trend =
                    points
                        |> List.map (\( meta, p ) -> ( p.x, p.y ))
                        |> Trend.Linear.quick

                xs =
                    points |> List.map (\( meta, p ) -> p.x)

                low =
                    xs |> List.minimum |> Maybe.withDefault 1

                high =
                    xs |> List.maximum |> Maybe.withDefault 10

                trendPoints : List DotDatum
                trendPoints =
                    trend
                        |> Result.map Trend.Linear.line
                        |> Result.map (lineToPoints ( low, high ))
                        |> Result.withDefault []
                        |> List.map (\p -> ( { family = "", display = "" }, p ))

                ( rSqAttrs, rSqHtml ) =
                    case Result.map Trend.Linear.goodnessOfFit trend of
                        Err err ->
                            ( [ HA.class "font-mono text-red-500" ]
                            , [ Html.text "Error: "
                              , Html.text <| formatTrendErr err
                              ]
                            )

                        Ok rSq ->
                            ( [ HA.class "font-mono text-blue-400 text-sm" ]
                            , [ Html.text "R^2="
                              , Html.text <| Round.round 2 rSq
                              ]
                            )

                rSqLabel =
                    C.htmlAt .max .min -75 20 rSqAttrs rSqHtml
            in
            C.chart
                [ CA.height 300
                , CA.width 300
                , CA.margin
                    { top = 25
                    , bottom = 30
                    , left = 45
                    , right = 10
                    }
                , CE.onMouseMove (OnHoverDot (Just key)) (CE.getNearest CI.dots)
                , CE.onMouseLeave (OnHoverDot Nothing [])
                ]
                [ C.xLabels
                    [ CA.withGrid
                    , CA.amount 3
                    , CA.format formatExp
                    ]
                , C.yLabels [ CA.withGrid, CA.amount 3 ]
                , C.series (Tuple.second >> .x)
                    [ C.interpolated (Tuple.second >> .y)
                        -- bg-blue-300
                        [ CA.color "var(--color-blue-300)", CA.width 2 ]
                        []
                        |> C.named "trendline"
                    ]
                    trendPoints
                , C.series (Tuple.second >> .x)
                    [ C.scatter (Tuple.second >> .y)
                        [ CA.opacity 0.5, CA.borderWidth 1 ]
                        |> C.variation
                            (\index ( metadata, datum ) ->
                                let
                                    color =
                                        metadata.family |> familyColor |> toCssColorVar
                                in
                                [ CA.color color, CA.border color ]
                            )
                        |> C.amongst hovered
                            (\_ -> [ CA.highlight 0.4, CA.opacity 1.0 ])
                    ]
                    points
                , C.each hovered
                    (\p dot ->
                        let
                            color =
                                CI.getColor dot

                            name =
                                CI.getData dot |> Tuple.first |> .display

                            x =
                                10 ^ (CI.getX dot - 6)
                        in
                        [ C.tooltip dot
                            [ CA.offset 0 ]
                            [ HA.style "color" color ]
                            [ Html.text name
                            , Html.text ": ("
                            , Html.text (Round.round 1 x)
                            , Html.text "M, "
                            , Html.text (Round.round 1 <| CI.getY dot)
                            , Html.text ")"
                            ]
                        ]
                    )
                , rSqLabel
                , C.labelAt
                    CA.middle
                    .min
                    [ CA.moveDown 35 ]
                    [ Svg.text "Params" ]
                , C.labelAt
                    .min
                    CA.middle
                    [ CA.moveLeft 35, CA.rotate 90 ]
                    [ Svg.text title ]
                ]


type alias PointMetadata =
    { family : String, display : String }


type alias BarDatum =
    ( PointMetadata, { score : Float } )


type alias DotDatum =
    ( PointMetadata, { x : Float, y : Float } )


rowToPointMetadata : TableRow -> PointMetadata
rowToPointMetadata row =
    { family = row.checkpoint.family, display = row.checkpoint.display }


toBarDatum : (TableRow -> Maybe Float) -> TableRow -> Maybe BarDatum
toBarDatum getter row =
    case getter row of
        Nothing ->
            Nothing

        Just score ->
            Just
                ( { family = row.checkpoint.family, display = row.checkpoint.display }
                , { score = score }
                )


viewScore : Maybe Float -> String
viewScore score =
    case score of
        Nothing ->
            "-"

        Just val ->
            val * 100 |> Round.round 1


getBenchmarkScore : String -> Set String -> TableRow -> Maybe Float
getBenchmarkScore task visibleKeys row =
    Dict.get task row.scores


viewBenchmarkScore : String -> Set String -> TableRow -> ( String, String )
viewBenchmarkScore task visibleKeys row =
    let
        score =
            getBenchmarkScore task visibleKeys row
    in
    ( "text-right font-mono tabular-nums"
    , score |> viewScore
    )


getMeanScore : Set String -> Set String -> TableRow -> Maybe Float
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


viewMeanScore : Set String -> Set String -> TableRow -> ( String, String )
viewMeanScore tasks columnsSelected row =
    ( "text-right font-mono tabular-nums"
    , getMeanScore tasks columnsSelected row |> viewScore
    )


getCheckpoint : Set String -> TableRow -> Maybe String
getCheckpoint cols row =
    Just row.checkpoint.display


viewCheckpoint : Set String -> TableRow -> ( String, String )
viewCheckpoint cols row =
    ( "text-left"
    , row.checkpoint.display
    )


getCheckpointParams : Set String -> TableRow -> Maybe Float
getCheckpointParams _ row =
    row.checkpoint.params |> toFloat |> Just


viewCheckpointParams : Set String -> TableRow -> ( String, String )
viewCheckpointParams columnsSelected row =
    ( "text-right"
    , toFloat row.checkpoint.params
        / (10 ^ 6)
        |> Round.round 0
    )


getCheckpointRelease : Set String -> TableRow -> Maybe Float
getCheckpointRelease _ row =
    row.checkpoint.release |> Maybe.map (Time.posixToMillis >> toFloat)


viewCheckpointRelease : Set String -> TableRow -> ( String, String )
viewCheckpointRelease _ row =
    ( "text-right"
    , row.checkpoint.release
        |> Maybe.map (formatTime Time.utc)
        |> Maybe.withDefault "unknown"
    )


getCheckpointResolution : Set String -> TableRow -> Maybe Float
getCheckpointResolution _ row =
    row.checkpoint.resolution |> toFloat |> Just


viewCheckpointResolution : Set String -> TableRow -> ( String, String )
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


formatExp : Float -> String
formatExp exp =
    case round exp of
        5 ->
            "100K"

        6 ->
            "1M"

        7 ->
            "10M"

        8 ->
            "100M"

        9 ->
            "1B"

        10 ->
            "10B"

        11 ->
            "100B"

        12 ->
            "1T"

        other ->
            String.fromFloat (10 ^ toFloat other)


formatRange : ( Int, Int ) -> String
formatRange ( min, max ) =
    if min == 0 then
        "<" ++ formatBigInt max

    else if max == upperLimit then
        formatBigInt min ++ "+"

    else
        formatBigInt min ++ "-" ++ formatBigInt max


formatBigInt : Int -> String
formatBigInt i =
    if i < 1000 then
        String.fromInt i

    else if i < 1000000 then
        String.fromInt (i // 1000) ++ "K"

    else if i < 1000000000 then
        String.fromInt (i // 1000000) ++ "M"

    else if i < 1000000000000 then
        String.fromInt (i // 1000000000) ++ "B"

    else
        String.fromInt i


formatTrendErr : Trend.Math.Error -> String
formatTrendErr err =
    case err of
        Trend.Math.NeedMoreValues min ->
            "Need at least " ++ String.fromInt min ++ " values."

        Trend.Math.AllZeros ->
            "All points are zeros."


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

        "OpenVision" ->
            -- accent-biobench-blue
            -- focus-visible:outline-biobench-blue
            "biobench-blue"

        "SigLIP" ->
            -- accent-biobench-cyan
            -- focus-visible:outline-biobench-cyan
            "biobench-cyan"

        "SigLIP2" ->
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

        "MetaCLIP 2" ->
            -- accent-biobench-purple
            -- focus-visible:outline-biobench-purple
            "biobench-purple"

        "Perception Encoder" ->
            -- accent-biobench-magenta
            -- focus-visible:outline-biobench-magenta
            "biobench-magenta"

        "DINOv3" ->
            -- accent-biobench-forest
            -- focus-visible:outline-biobench-forest
            "biobench-forest"

        "ViT" ->
            -- accent-biobench-slate
            -- focus-visible:outline-biobench-slate
            "biobench-slate"

        _ ->
            -- accent-biobench-black
            -- focus-visible:outline-biobench-black
            "biobench-black"


upperLimit : Int
upperLimit =
    round (10 ^ 16)


allParamRanges : List ( Int, Int )
allParamRanges =
    [ ( 0, 50000000 )
    , ( 50000000, 100000000 )
    , ( 100000000, 400000000 )
    , ( 400000000, 1000000000 )
    , ( 1000000000, 2000000000 )
    , ( 2000000000, upperLimit )
    ]


paramRangesMatch : Set ( Int, Int ) -> Int -> Bool
paramRangesMatch ranges params =
    List.any (rangeMatches params) (Set.toList ranges)


rangeMatches : Int -> ( Int, Int ) -> Bool
rangeMatches x ( low, high ) =
    low < x && x < high



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
            , { key = "inat21", display = "iNat21", format = viewBenchmarkScore "inat21", sortType = SortNumeric (getBenchmarkScore "inat21"), barchart = True, immediatelyVisible = False }
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


getScores : Checkpoint -> List Score -> Dict String Float
getScores checkpoint scores =
    scores
        |> List.filter (\score -> score.checkpoint == checkpoint.name)
        |> List.map (\score -> ( score.task, score.mean ))
        |> Dict.fromList


getWinners : Checkpoint -> List Best -> Set String
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
    , ties : Set String
    }


bestDecoder : D.Decoder Best
bestDecoder =
    D.map2 Best
        (D.field "task" D.string)
        (D.field "ties"
            (D.map Set.fromList (D.list D.string))
        )


viewCheckboxFieldset : Fieldset -> Bool -> Set a -> List b -> (b -> Html Msg) -> Html Msg
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
                    [ Html.text (formatFieldset fieldset)
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
        , HE.custom "change" (D.succeed { message = msg, stopPropagation = True, preventDefault = True })
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
        , HE.custom "change" (D.succeed { message = ToggleFamily family, stopPropagation = True, preventDefault = True })
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


formatFieldset : Fieldset -> String
formatFieldset fieldset =
    case fieldset of
        ColumnsFieldset ->
            "Columns"

        FamiliesFieldset ->
            "Model Families"

        ParamRangesFieldset ->
            "# Parameters"

        ResolutionsFieldset ->
            "Resolution"



-- CSS


radioClass =
    "accent-blue-500 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-blue-500"


checkboxClass =
    "accent-blue-500 cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-blue-500"



-- CHART HELPERS


toDot : String -> (Float -> Float) -> String -> (Float -> Float) -> ( PointMetadata, Dict String Float ) -> Maybe DotDatum
toDot fieldX fnX fieldY fnY ( meta, scores ) =
    case ( Dict.get fieldX scores, Dict.get fieldY scores ) of
        ( Just x, Just y ) ->
            Just ( meta, { x = fnX x, y = fnY y } )

        ( _, _ ) ->
            Nothing


filterHovered : String -> Maybe String -> List (CI.One data x) -> List (CI.One data x)
filterHovered key hoveredKey hoveredItems =
    case hoveredKey of
        Just incoming ->
            if key == incoming then
                hoveredItems

            else
                []

        Nothing ->
            []
