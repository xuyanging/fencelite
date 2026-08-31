// arrow-sidecar/cli.mjs
import { readFile } from "node:fs/promises";

// arrow-sidecar/vendor/scene-spatial-index.ts
var TARGET_BUCKET_LOAD = 18;
var MAX_AXIS_CELLS = 2048;
var MAX_CELLS_PER_ITEM = 64;
var FULL_SCAN_CELL_THRESHOLD = 16384;
var validBounds = (bounds) => Boolean(
  bounds && Number.isFinite(bounds.minX) && Number.isFinite(bounds.minY) && Number.isFinite(bounds.maxX) && Number.isFinite(bounds.maxY) && bounds.maxX >= bounds.minX && bounds.maxY >= bounds.minY
);
var intersects = (left, right) => left.maxX >= right.minX && left.minX <= right.maxX && left.maxY >= right.minY && left.minY <= right.maxY;
var unionSceneBounds = (scene) => {
  const page = validBounds(scene.pageBounds) ? scene.pageBounds : void 0;
  const parsed = validBounds(scene.bounds) ? scene.bounds : void 0;
  const result = page ? { ...page } : parsed ? { ...parsed } : { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  for (const op of scene.ops) {
    if (!validBounds(op.bounds)) continue;
    result.minX = Math.min(result.minX, op.bounds.minX);
    result.minY = Math.min(result.minY, op.bounds.minY);
    result.maxX = Math.max(result.maxX, op.bounds.maxX);
    result.maxY = Math.max(result.maxY, op.bounds.maxY);
  }
  if (result.maxX === result.minX) result.maxX += 1;
  if (result.maxY === result.minY) result.maxY += 1;
  return result;
};
var SceneSpatialIndex = class {
  constructor(scene) {
    this.buckets = /* @__PURE__ */ new Map();
    this.largeIndices = [];
    this.allIndices = [];
    this.seenGeneration = 0;
    this.scene = scene;
    this.bounds = unionSceneBounds(scene);
    const width = Math.max(1e-6, this.bounds.maxX - this.bounds.minX);
    const height = Math.max(1e-6, this.bounds.maxY - this.bounds.minY);
    const targetCells = Math.max(1, Math.ceil(scene.ops.length / TARGET_BUCKET_LOAD));
    const aspect = width / height;
    this.columns = Math.max(
      1,
      Math.min(MAX_AXIS_CELLS, Math.ceil(Math.sqrt(targetCells * aspect)))
    );
    this.rows = Math.max(
      1,
      Math.min(MAX_AXIS_CELLS, Math.ceil(targetCells / this.columns))
    );
    this.cellWidth = width / this.columns;
    this.cellHeight = height / this.rows;
    this.seen = new Uint32Array(scene.ops.length);
    let entries = 0;
    scene.ops.forEach((op, index) => {
      if (!validBounds(op.bounds)) return;
      this.allIndices.push(index);
      const range = this.cellRange(op.bounds);
      const cells = (range.maxColumn - range.minColumn + 1) * (range.maxRow - range.minRow + 1);
      if (cells > MAX_CELLS_PER_ITEM) {
        this.largeIndices.push(index);
        return;
      }
      for (let row = range.minRow; row <= range.maxRow; row += 1) {
        for (let column = range.minColumn; column <= range.maxColumn; column += 1) {
          const key = row * this.columns + column;
          const bucket = this.buckets.get(key);
          if (bucket) bucket.push(index);
          else this.buckets.set(key, [index]);
          entries += 1;
        }
      }
    });
    this.entryCount = entries + this.largeIndices.length;
  }
  columnAt(x) {
    return Math.max(0, Math.min(
      this.columns - 1,
      Math.floor((x - this.bounds.minX) / this.cellWidth)
    ));
  }
  rowAt(y) {
    return Math.max(0, Math.min(
      this.rows - 1,
      Math.floor((y - this.bounds.minY) / this.cellHeight)
    ));
  }
  cellRange(bounds) {
    return {
      minColumn: this.columnAt(bounds.minX),
      maxColumn: this.columnAt(bounds.maxX),
      minRow: this.rowAt(bounds.minY),
      maxRow: this.rowAt(bounds.maxY)
    };
  }
  nextGeneration() {
    this.seenGeneration = this.seenGeneration + 1 >>> 0;
    if (this.seenGeneration === 0) {
      this.seen.fill(0);
      this.seenGeneration = 1;
    }
    return this.seenGeneration;
  }
  queryBounds(bounds) {
    if (!validBounds(bounds) || !intersects(bounds, this.bounds)) return [];
    const range = this.cellRange(bounds);
    const cellCount = (range.maxColumn - range.minColumn + 1) * (range.maxRow - range.minRow + 1);
    if (cellCount >= FULL_SCAN_CELL_THRESHOLD) {
      return this.allIndices.filter((index) => intersects(this.scene.ops[index].bounds, bounds));
    }
    const generation = this.nextGeneration();
    const result = [];
    const append = (index) => {
      if (this.seen[index] === generation) return;
      this.seen[index] = generation;
      const opBounds = this.scene.ops[index]?.bounds;
      if (validBounds(opBounds) && intersects(opBounds, bounds)) result.push(index);
    };
    for (let row = range.minRow; row <= range.maxRow; row += 1) {
      for (let column = range.minColumn; column <= range.maxColumn; column += 1) {
        const bucket = this.buckets.get(row * this.columns + column);
        if (bucket) bucket.forEach(append);
      }
    }
    this.largeIndices.forEach(append);
    result.sort((left, right) => left - right);
    return result;
  }
  queryPoint(x, y, tolerance = 0) {
    const safeTolerance = Number.isFinite(tolerance) ? Math.max(0, tolerance) : 0;
    return this.queryBounds({
      minX: x - safeTolerance,
      minY: y - safeTolerance,
      maxX: x + safeTolerance,
      maxY: y + safeTolerance
    });
  }
};
var sceneSpatialIndexCache = /* @__PURE__ */ new WeakMap();
var getSceneSpatialIndex = (scene) => {
  let index = sceneSpatialIndexCache.get(scene);
  if (!index) {
    index = new SceneSpatialIndex(scene);
    sceneSpatialIndexCache.set(scene, index);
  }
  return index;
};

// arrow-sidecar/vendor/callout-detection.ts
var CALLOUT_DETECTION_VERSION = "callout-vector-v19";
var DEFAULT_CALLOUT_DETECTION_OPTIONS = {
  textLineGapScale: 2.8,
  leaderRootDistanceScale: 2.5,
  leaderConnectionScale: 1.25,
  maxLocalPaintGap: 20,
  formalConfidence: 0.72,
  reviewConfidence: 0.46
};
var finiteBounds = (bounds) => ({
  minX: Math.min(bounds.minX, bounds.maxX),
  minY: Math.min(bounds.minY, bounds.maxY),
  maxX: Math.max(bounds.minX, bounds.maxX),
  maxY: Math.max(bounds.minY, bounds.maxY)
});
var carrierBoundsForCluster = (scene, cluster) => {
  const symbolCarrier = cluster.carrierKind === "symbol" || cluster.carrierKind === "text-symbol";
  if (!symbolCarrier) return cluster.bounds;
  if (cluster.carrierBounds) return finiteBounds(cluster.carrierBounds);
  if (cluster.symbolOpIndex !== void 0) return finiteBounds(scene.ops[cluster.symbolOpIndex].bounds);
  return cluster.bounds;
};
var unionBounds = (left, right) => ({
  minX: Math.min(left.minX, right.minX),
  minY: Math.min(left.minY, right.minY),
  maxX: Math.max(left.maxX, right.maxX),
  maxY: Math.max(left.maxY, right.maxY)
});
var boundsWidth = (bounds) => Math.max(0, bounds.maxX - bounds.minX);
var boundsHeight = (bounds) => Math.max(0, bounds.maxY - bounds.minY);
var boundsDiagonal = (bounds) => Math.hypot(boundsWidth(bounds), boundsHeight(bounds));
var boundsGap = (left, right) => Math.hypot(
  Math.max(0, left.minX - right.maxX, right.minX - left.maxX),
  Math.max(0, left.minY - right.maxY, right.minY - left.maxY)
);
var pointDistance = (left, right) => Math.hypot(left.x - right.x, left.y - right.y);
var pointToSegment = (point, start, end) => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= 1e-12) return pointDistance(point, start);
  const offset = clamp01(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared);
  return pointDistance(point, { x: start.x + offset * dx, y: start.y + offset * dy });
};
var pointToBounds = (point, bounds) => Math.hypot(
  Math.max(0, bounds.minX - point.x, point.x - bounds.maxX),
  Math.max(0, bounds.minY - point.y, point.y - bounds.maxY)
);
var intersects2 = (left, right) => left.minX <= right.maxX && left.maxX >= right.minX && left.minY <= right.maxY && left.maxY >= right.minY;
var boundsArea = (bounds) => Math.max(0, bounds.maxX - bounds.minX) * Math.max(0, bounds.maxY - bounds.minY);
var boundsOverlapFraction = (bounds, roi) => {
  const intersection = {
    minX: Math.max(bounds.minX, roi.minX),
    minY: Math.max(bounds.minY, roi.minY),
    maxX: Math.min(bounds.maxX, roi.maxX),
    maxY: Math.min(bounds.maxY, roi.maxY)
  };
  return boundsArea(intersection) / Math.max(1e-9, boundsArea(bounds));
};
var contains = (outer, inner, padding = 0) => outer.minX - padding <= inner.minX && outer.minY - padding <= inner.minY && outer.maxX + padding >= inner.maxX && outer.maxY + padding >= inner.maxY;
var clamp01 = (value) => Math.max(0, Math.min(1, value));
var pointToPath = (point, op) => {
  let first = null;
  let previous = null;
  let distance = Number.POSITIVE_INFINITY;
  for (const segment of op.segments) {
    if (segment.kind === "move") {
      first = { x: segment.x, y: segment.y };
      previous = first;
    } else if (segment.kind === "line" || segment.kind === "curve") {
      const next = { x: segment.x, y: segment.y };
      if (previous) distance = Math.min(distance, pointToSegment(point, previous, next));
      previous = next;
    } else if (segment.kind === "close" && previous && first) {
      distance = Math.min(distance, pointToSegment(point, previous, first));
      previous = first;
    }
  }
  return distance;
};
var endpointContactThreshold = (pageDiagonal, carrierScale, connectionScale, pathLineWidth, markerOrPathScale = 0) => Math.min(
  pageDiagonal * 6e-4,
  Math.max(
    pageDiagonal * 4e-5,
    carrierScale * connectionScale * 0.1,
    Math.max(0, pathLineWidth) * 2.5,
    markerOrPathScale * 0.18
  )
);
var markerTouchesEndpoint = (marker, endpoints, pageDiagonal, carrierScale, connectionScale, pathLineWidth) => {
  const threshold = endpointContactThreshold(
    pageDiagonal,
    carrierScale,
    connectionScale,
    Math.max(pathLineWidth, marker.lineWidth),
    boundsDiagonal(marker.bounds)
  );
  const mergeDistance = Math.max(pageDiagonal * 1e-5, threshold * 0.25);
  const terminals = endpoints.reduce((unique, endpoint) => {
    if (!unique.some((candidate) => pointDistance(candidate, endpoint) <= mergeDistance)) {
      unique.push(endpoint);
    }
    return unique;
  }, []);
  return terminals.filter((endpoint) => pointToPath(endpoint, marker) <= threshold).length === 1;
};
var markerBundleTouchesEndpoint = (scene, markerOps, endpoints, pageDiagonal, pathLineWidth) => {
  const markers = markerOps.map((opIndex) => scene.ops[opIndex]).filter((op) => op?.kind === "path");
  if (!markers.length) return false;
  const markerScale = Math.max(...markers.map((marker) => boundsDiagonal(finiteBounds(marker.bounds))));
  const threshold = endpointContactThreshold(
    pageDiagonal,
    markerScale,
    1,
    Math.max(pathLineWidth, ...markers.map((marker) => marker.lineWidth)),
    markerScale
  );
  const mergeDistance = Math.max(pageDiagonal * 1e-5, threshold * 0.25);
  const terminals = endpoints.reduce((unique, endpoint) => {
    if (!unique.some((candidate) => pointDistance(candidate, endpoint) <= mergeDistance)) unique.push(endpoint);
    return unique;
  }, []);
  return terminals.filter((endpoint) => Math.min(...markers.map((marker) => pointToPath(endpoint, marker))) <= threshold).length === 1;
};
var pathEndpoints = (op) => {
  let first = null;
  let last = null;
  for (const segment of op.segments) {
    if (segment.kind === "move" || segment.kind === "line") {
      const point = { x: segment.x, y: segment.y };
      first ??= point;
      last = point;
    } else if (segment.kind === "curve") {
      const point = { x: segment.x, y: segment.y };
      first ??= point;
      last = point;
    }
  }
  return first && last ? [first, last] : null;
};
var pathVertices = (op) => op.segments.flatMap((segment) => segment.kind === "move" || segment.kind === "line" || segment.kind === "curve" ? [{ x: segment.x, y: segment.y }] : []);
var terminalOutwardUnit = (scene, pathOps, root, leaderEnd, pageDiagonal) => {
  const endpointTolerance = Math.max(1e-8, pageDiagonal * 1e-5);
  let best = null;
  for (const opIndex of pathOps) {
    const op = scene.ops[opIndex];
    if (op?.kind !== "path") continue;
    const vertices = pathVertices(op);
    if (vertices.length < 2) continue;
    const endpointPairs = [
      [vertices[0], vertices[1]],
      [vertices.at(-1), vertices.at(-2)]
    ];
    for (const [endpoint, inward] of endpointPairs) {
      const dx2 = endpoint.x - inward.x;
      const dy2 = endpoint.y - inward.y;
      const length2 = Math.hypot(dx2, dy2);
      if (length2 <= endpointTolerance) continue;
      const candidate = { distance: pointDistance(endpoint, leaderEnd), dx: dx2 / length2, dy: dy2 / length2 };
      if (!best || candidate.distance < best.distance) best = candidate;
    }
  }
  if (best && best.distance <= pageDiagonal * 3e-3) return { x: best.dx, y: best.dy };
  const dx = leaderEnd.x - root.x;
  const dy = leaderEnd.y - root.y;
  const length = Math.hypot(dx, dy);
  return length > endpointTolerance ? { x: dx / length, y: dy / length } : { x: 0, y: 0 };
};
var triangleArrowApex = (scene, markerOps, pathOps, root, leaderEnd, pageDiagonal) => {
  if (!markerOps.length) return null;
  const markerPaths = markerOps.map((opIndex) => scene.ops[opIndex]).filter((op) => op?.kind === "path");
  if (!markerPaths.length) return null;
  const markerSpan = Math.max(...markerPaths.map((op) => boundsDiagonal(finiteBounds(op.bounds))));
  const dedupeThreshold = Math.max(1e-8, pageDiagonal * 2e-6, markerSpan * 0.01);
  const vertices = [];
  for (const op of markerPaths) {
    for (const vertex of pathVertices(op)) {
      if (!vertices.some((owned) => pointDistance(owned, vertex) <= dedupeThreshold)) {
        vertices.push(vertex);
      }
    }
  }
  if (vertices.length !== 3) return null;
  const twiceArea = Math.abs(
    (vertices[1].x - vertices[0].x) * (vertices[2].y - vertices[0].y) - (vertices[1].y - vertices[0].y) * (vertices[2].x - vertices[0].x)
  );
  if (twiceArea <= Math.max(dedupeThreshold ** 2, markerSpan ** 2 * 5e-3)) return null;
  const connectionThreshold = Math.max(pageDiagonal * 35e-5, markerSpan * 0.35);
  if (Math.min(...markerPaths.map((op) => pointToPath(leaderEnd, op))) > connectionThreshold) return null;
  const outward = terminalOutwardUnit(scene, pathOps, root, leaderEnd, pageDiagonal);
  if (Math.hypot(outward.x, outward.y) < 0.5) return null;
  const ranked = vertices.map((vertex) => {
    const vx = vertex.x - leaderEnd.x;
    const vy = vertex.y - leaderEnd.y;
    return {
      vertex,
      forward: vx * outward.x + vy * outward.y,
      lateral: Math.abs(vx * outward.y - vy * outward.x)
    };
  }).sort((left, right) => right.forward - left.forward || left.lateral - right.lateral);
  const edges = [
    { length: pointDistance(vertices[0], vertices[1]), opposite: vertices[2] },
    { length: pointDistance(vertices[1], vertices[2]), opposite: vertices[0] },
    { length: pointDistance(vertices[2], vertices[0]), opposite: vertices[1] }
  ].sort((left, right) => left.length - right.length);
  if (edges[0].length <= edges[1].length * 0.5 && edges[1].length >= edges[2].length * 0.9) {
    return edges[0].opposite;
  }
  const margin = ranked[0].forward - ranked[1].forward;
  if (ranked[0].forward < -connectionThreshold || margin < Math.max(pageDiagonal * 2e-5, markerSpan * 0.05)) return null;
  return ranked[0].vertex;
};
var spatialTriangleMarkerMatches = (scene, markerOpIndex, pathOps, root, leaderEnd, routeLength, pageDiagonal) => {
  const marker = scene.ops[markerOpIndex];
  if (marker?.kind !== "path") return false;
  const markerSpan = boundsDiagonal(finiteBounds(marker.bounds));
  if (routeLength < Math.max(markerSpan * 2.5, pageDiagonal * 2e-3)) return false;
  const dedupeThreshold = Math.max(1e-8, pageDiagonal * 2e-6, markerSpan * 0.01);
  const vertices = pathVertices(marker).reduce((unique, vertex) => {
    if (!unique.some((owned) => pointDistance(owned, vertex) <= dedupeThreshold)) unique.push(vertex);
    return unique;
  }, []);
  if (vertices.length !== 3) return false;
  const twiceArea = Math.abs(
    (vertices[1].x - vertices[0].x) * (vertices[2].y - vertices[0].y) - (vertices[1].y - vertices[0].y) * (vertices[2].x - vertices[0].x)
  );
  if (twiceArea <= Math.max(dedupeThreshold ** 2, markerSpan ** 2 * 5e-3)) return false;
  const edges = [
    { start: vertices[0], end: vertices[1], length: pointDistance(vertices[0], vertices[1]), opposite: vertices[2] },
    { start: vertices[1], end: vertices[2], length: pointDistance(vertices[1], vertices[2]), opposite: vertices[0] },
    { start: vertices[2], end: vertices[0], length: pointDistance(vertices[2], vertices[0]), opposite: vertices[1] }
  ].sort((left, right) => left.length - right.length);
  const base = edges[0];
  if (base.length > edges[1].length * 0.5 || edges[1].length < edges[2].length * 0.9) return false;
  const baseMidpoint = {
    x: (base.start.x + base.end.x) / 2,
    y: (base.start.y + base.end.y) / 2
  };
  const axisDx = base.opposite.x - baseMidpoint.x;
  const axisDy = base.opposite.y - baseMidpoint.y;
  const axisLength = Math.hypot(axisDx, axisDy);
  if (axisLength <= dedupeThreshold) return false;
  const outward = terminalOutwardUnit(scene, pathOps, root, leaderEnd, pageDiagonal);
  return (axisDx * outward.x + axisDy * outward.y) / axisLength >= 0.8;
};
var resolveLeaderTarget = (scene, leader, root, connection, pageDiagonal) => {
  const markerOps = [...leader.markerOps];
  const arrowheadOp = leader.arrowheadOps[0];
  if (leader.markerKind === "open-marker") {
    const tip = leader.openMarkerTip && pointDistance(leader.openMarkerTip, connection) <= pageDiagonal * 2e-3 ? leader.openMarkerTip : connection;
    return { ...tip, connection: { ...connection }, markerOps, terminalKind: "open-marker-contact", arrowheadOp };
  }
  if (leader.markerKind === "filled-arrow" || leader.markerKind === "split-arrow") {
    const apex = triangleArrowApex(scene, markerOps, leader.pathOps, root, connection, pageDiagonal);
    if (apex) return { ...apex, connection: { ...connection }, markerOps, terminalKind: "arrow-apex", arrowheadOp };
  }
  if (markerOps.length) {
    return { ...connection, connection: { ...connection }, markerOps, terminalKind: "marker-contact", arrowheadOp };
  }
  return { ...connection, connection: { ...connection }, markerOps: [], terminalKind: "free-end", arrowheadOp };
};
var textScale = (op) => Math.max(
  1,
  Math.min(
    Math.max(1, boundsWidth(op.bounds)),
    Math.max(1, boundsHeight(op.bounds))
  )
);
var orientation = (op) => Math.atan2(op.matrix[1], op.matrix[0]);
var orientationDelta = (left, right) => {
  const raw = Math.abs(orientation(left) - orientation(right)) % Math.PI;
  return Math.min(raw, Math.PI - raw);
};
var clusterTextOps = (scene, pageDiagonal, options) => {
  const textEntries = scene.ops.map((op, index) => ({ op, index })).filter((entry) => entry.op.kind === "text").filter((entry) => entry.op.text.trim().length > 0);
  const clusters = [];
  for (const entry of textEntries) {
    const previous = clusters.at(-1);
    const previousOp = previous?.ops.at(-1);
    const scale = textScale(entry.op);
    const gapThreshold = Math.min(
      pageDiagonal * 0.018,
      Math.max(scale, previous?.scale ?? scale) * options.textLineGapScale
    );
    const joins = Boolean(
      previous && previousOp && entry.index - previous.opIndices.at(-1) <= 2 && orientationDelta(previousOp, entry.op) <= 0.12 && boundsGap(previous.bounds, finiteBounds(entry.op.bounds)) <= gapThreshold
    );
    if (joins && previous) {
      previous.opIndices.push(entry.index);
      previous.ops.push(entry.op);
      previous.bounds = unionBounds(previous.bounds, finiteBounds(entry.op.bounds));
      previous.text += `
${entry.op.text.trim()}`;
      previous.scale = Math.max(previous.scale, scale);
    } else {
      clusters.push({
        opIndices: [entry.index],
        ops: [entry.op],
        bounds: finiteBounds(entry.op.bounds),
        text: entry.op.text.trim(),
        scale,
        source: "decoded"
      });
    }
  }
  return clusters;
};
var isCompactArrowhead = (op, pageDiagonal) => Boolean(op?.kind === "path" && op.fill && op.stroke && op.segments.length >= 4 && op.segments.length <= 6 && boundsDiagonal(op.bounds) >= pageDiagonal * 35e-5 && boundsDiagonal(op.bounds) <= pageDiagonal * 0.018);
var isCompactArrowPart = (op, pageDiagonal) => Boolean(op?.kind === "path" && (op.fill || op.stroke) && // A fill-only triangle can legally omit `close`: m l l f is three parsed
// segments even though PDF filling closes it implicitly. Taylor 3-12 uses
// exactly this fill-3 + stroke-4 arrow packet.
op.segments.length >= 3 && op.segments.length <= 6 && boundsDiagonal(op.bounds) >= pageDiagonal * 35e-5 && boundsDiagonal(op.bounds) <= pageDiagonal * 0.018);
var splitArrowBefore = (scene, opIndex, pageDiagonal) => {
  const first = scene.ops[opIndex - 2];
  const second = scene.ops[opIndex - 1];
  if (!isCompactArrowPart(first, pageDiagonal) || !isCompactArrowPart(second, pageDiagonal)) return [];
  const complementaryPaint = first.fill && !first.stroke && second.stroke && !second.fill || first.stroke && !first.fill && second.fill && !second.stroke;
  const sameShape = boundsGap(finiteBounds(first.bounds), finiteBounds(second.bounds)) <= pageDiagonal * 5e-4 && Math.abs(boundsDiagonal(first.bounds) - boundsDiagonal(second.bounds)) <= pageDiagonal * 1e-3;
  return complementaryPaint && sameShape ? [opIndex - 2, opIndex - 1] : [];
};
var compoundEndpointMarkerBefore = (scene, opIndex, endpoints, pageDiagonal, segmentation) => {
  const pairs = [];
  const leaderGroup = groupForOp(segmentation, opIndex);
  for (let pairEnd = opIndex - 1; pairEnd >= 1 && pairs.length < 8; pairEnd -= 2) {
    const fill = scene.ops[pairEnd - 1];
    const stroke = scene.ops[pairEnd];
    if (fill?.kind !== "path" || stroke?.kind !== "path" || !fill.fill || fill.stroke || !stroke.stroke || stroke.fill || groupForOp(segmentation, pairEnd - 1) !== leaderGroup || groupForOp(segmentation, pairEnd) !== leaderGroup || fill.segments.length < 3 || fill.segments.length > 16 || stroke.segments.length < 3 || stroke.segments.length > 16) break;
    const fillDiagonal = boundsDiagonal(fill.bounds);
    const strokeDiagonal = boundsDiagonal(stroke.bounds);
    if (fillDiagonal < pageDiagonal * 25e-5 || fillDiagonal > pageDiagonal * 0.012 || strokeDiagonal < pageDiagonal * 25e-5 || strokeDiagonal > pageDiagonal * 0.012 || boundsGap(finiteBounds(fill.bounds), finiteBounds(stroke.bounds)) > pageDiagonal * 5e-4 || boundsOverlapFraction(finiteBounds(fill.bounds), finiteBounds(stroke.bounds)) < 0.7 || boundsOverlapFraction(finiteBounds(stroke.bounds), finiteBounds(fill.bounds)) < 0.7 || Math.abs(fillDiagonal - strokeDiagonal) > pageDiagonal * 1e-3) break;
    pairs.unshift([pairEnd - 1, pairEnd]);
  }
  const markerOps = pairs.flat();
  if (markerOps.length < 6) return [];
  const markerBounds = markerOps.slice(1).reduce(
    (combined, index) => unionBounds(combined, finiteBounds(scene.ops[index].bounds)),
    finiteBounds(scene.ops[markerOps[0]].bounds)
  );
  if (boundsDiagonal(markerBounds) > pageDiagonal * 0.015) return [];
  const endpointDistances = endpoints.map((endpoint) => Math.min(...markerOps.map((index) => {
    const marker = scene.ops[index];
    return marker.kind === "path" ? pointToPath(endpoint, marker) : Number.POSITIVE_INFINITY;
  })));
  const touches = endpointDistances.filter((distance) => distance <= pageDiagonal * 2e-3).length;
  if (touches !== 1 || pointDistance(endpoints[0], endpoints[1]) < boundsDiagonal(markerBounds) * 2.5) return [];
  return markerOps;
};
var isOpenEndpointMarkerPart = (op, pageDiagonal) => Boolean(op?.kind === "path" && op.stroke && !op.fill && op.segments.length === 2 && op.segments[0]?.kind === "move" && op.segments[1]?.kind === "curve" && boundsDiagonal(op.bounds) >= pageDiagonal * 35e-5 && boundsDiagonal(op.bounds) <= pageDiagonal * 6e-3);
var openEndpointMarkerBefore = (scene, opIndex, pageDiagonal) => {
  const first = scene.ops[opIndex - 2];
  const second = scene.ops[opIndex - 1];
  const leader = scene.ops[opIndex];
  if (!isOpenEndpointMarkerPart(first, pageDiagonal) || !isOpenEndpointMarkerPart(second, pageDiagonal) || leader?.kind !== "path" || !leader.stroke || leader.fill || leader.segments.length < 2 || leader.segments.length > 4) return null;
  const firstEnds = pathEndpoints(first);
  const secondEnds = pathEndpoints(second);
  const leaderEnds = pathEndpoints(leader);
  if (!firstEnds || !secondEnds || !leaderEnds) return null;
  const joins = [
    { gap: pointDistance(firstEnds[0], secondEnds[0]), firstOuter: firstEnds[1], secondOuter: secondEnds[1], firstJoin: firstEnds[0], secondJoin: secondEnds[0] },
    { gap: pointDistance(firstEnds[0], secondEnds[1]), firstOuter: firstEnds[1], secondOuter: secondEnds[0], firstJoin: firstEnds[0], secondJoin: secondEnds[1] },
    { gap: pointDistance(firstEnds[1], secondEnds[0]), firstOuter: firstEnds[0], secondOuter: secondEnds[1], firstJoin: firstEnds[1], secondJoin: secondEnds[0] },
    { gap: pointDistance(firstEnds[1], secondEnds[1]), firstOuter: firstEnds[0], secondOuter: secondEnds[0], firstJoin: firstEnds[1], secondJoin: secondEnds[1] }
  ].sort((left, right) => left.gap - right.gap);
  const join = joins[0];
  const connectionThreshold = pageDiagonal * 12e-4;
  if (join.gap > connectionThreshold) return null;
  const joint = {
    x: (join.firstJoin.x + join.secondJoin.x) / 2,
    y: (join.firstJoin.y + join.secondJoin.y) / 2
  };
  if (Math.min(pointDistance(joint, leaderEnds[0]), pointDistance(joint, leaderEnds[1])) > connectionThreshold) return null;
  const firstDx = join.firstOuter.x - joint.x;
  const firstDy = join.firstOuter.y - joint.y;
  const secondDx = join.secondOuter.x - joint.x;
  const secondDy = join.secondOuter.y - joint.y;
  const firstLength = Math.hypot(firstDx, firstDy);
  const secondLength = Math.hypot(secondDx, secondDy);
  if (firstLength < pageDiagonal * 2e-4 || secondLength < pageDiagonal * 2e-4) return null;
  const opposed = (firstDx * secondDx + firstDy * secondDy) / (firstLength * secondLength);
  if (opposed > -0.4) return null;
  const markerBounds = unionBounds(finiteBounds(first.bounds), finiteBounds(second.bounds));
  const markerSpan = boundsDiagonal(markerBounds);
  if (pointDistance(leaderEnds[0], leaderEnds[1]) < markerSpan * 2.5) return null;
  return { ops: [opIndex - 2, opIndex - 1], tip: joint };
};
var foldedOpenEndpointPacketBefore = (scene, stemIndex, pageDiagonal, segmentation) => {
  const authoredTextOp = stemIndex - 4;
  const branchIndex = stemIndex - 3;
  const firstWingIndex = stemIndex - 2;
  const secondWingIndex = stemIndex - 1;
  const text = scene.ops[authoredTextOp];
  const branch = scene.ops[branchIndex];
  const firstWing = scene.ops[firstWingIndex];
  const secondWing = scene.ops[secondWingIndex];
  const stem = scene.ops[stemIndex];
  if (text?.kind !== "text" || !text.text.trim() || branch?.kind !== "path" || !branch.stroke || branch.fill || branch.segments.length !== 2 || firstWing?.kind !== "path" || !firstWing.stroke || firstWing.fill || firstWing.segments.length !== 2 || secondWing?.kind !== "path" || !secondWing.stroke || secondWing.fill || secondWing.segments.length !== 2 || stem?.kind !== "path" || !stem.stroke || stem.fill || stem.segments.length !== 2 || firstWing.segments[0]?.kind !== "move" || firstWing.segments[1]?.kind !== "line" || secondWing.segments[0]?.kind !== "move" || secondWing.segments[1]?.kind !== "line") return null;
  const group = groupForOp(segmentation, stemIndex);
  if ([authoredTextOp, branchIndex, firstWingIndex, secondWingIndex].some((opIndex) => groupForOp(segmentation, opIndex) !== group)) return null;
  const branchEnds = pathEndpoints(branch);
  const firstWingEnds = pathEndpoints(firstWing);
  const secondWingEnds = pathEndpoints(secondWing);
  const stemEnds = pathEndpoints(stem);
  if (!branchEnds || !firstWingEnds || !secondWingEnds || !stemEnds) return null;
  const joins = [
    { gap: pointDistance(firstWingEnds[0], secondWingEnds[0]), firstOuter: firstWingEnds[1], secondOuter: secondWingEnds[1], firstJoin: firstWingEnds[0], secondJoin: secondWingEnds[0] },
    { gap: pointDistance(firstWingEnds[0], secondWingEnds[1]), firstOuter: firstWingEnds[1], secondOuter: secondWingEnds[0], firstJoin: firstWingEnds[0], secondJoin: secondWingEnds[1] },
    { gap: pointDistance(firstWingEnds[1], secondWingEnds[0]), firstOuter: firstWingEnds[0], secondOuter: secondWingEnds[1], firstJoin: firstWingEnds[1], secondJoin: secondWingEnds[0] },
    { gap: pointDistance(firstWingEnds[1], secondWingEnds[1]), firstOuter: firstWingEnds[0], secondOuter: secondWingEnds[0], firstJoin: firstWingEnds[1], secondJoin: secondWingEnds[1] }
  ].sort((left, right) => left.gap - right.gap);
  const join = joins[0];
  const connectionThreshold = pageDiagonal * 12e-4;
  if (join.gap > connectionThreshold) return null;
  const tip = {
    x: (join.firstJoin.x + join.secondJoin.x) / 2,
    y: (join.firstJoin.y + join.secondJoin.y) / 2
  };
  const firstWingVector = { x: join.firstOuter.x - tip.x, y: join.firstOuter.y - tip.y };
  const secondWingVector = { x: join.secondOuter.x - tip.x, y: join.secondOuter.y - tip.y };
  const firstWingLength = Math.hypot(firstWingVector.x, firstWingVector.y);
  const secondWingLength = Math.hypot(secondWingVector.x, secondWingVector.y);
  const markerBounds = unionBounds(finiteBounds(firstWing.bounds), finiteBounds(secondWing.bounds));
  if (firstWingLength < pageDiagonal * 2e-4 || secondWingLength < pageDiagonal * 2e-4 || boundsDiagonal(markerBounds) > pageDiagonal * 6e-3) return null;
  const wingDot = (firstWingVector.x * secondWingVector.x + firstWingVector.y * secondWingVector.y) / (firstWingLength * secondWingLength);
  const wingCross = Math.abs(firstWingVector.x * secondWingVector.y - firstWingVector.y * secondWingVector.x) / (firstWingLength * secondWingLength);
  if (wingDot > 0.8 || wingCross < 0.25) return null;
  const branchTipDistances = branchEnds.map((point) => pointDistance(point, tip));
  const branchTipIndex = branchTipDistances[0] <= branchTipDistances[1] ? 0 : 1;
  if (branchTipDistances[branchTipIndex] > connectionThreshold) return null;
  const elbow = branchEnds[branchTipIndex === 0 ? 1 : 0];
  const stemElbowDistances = stemEnds.map((point) => pointDistance(point, elbow));
  const stemElbowIndex = stemElbowDistances[0] <= stemElbowDistances[1] ? 0 : 1;
  if (stemElbowDistances[stemElbowIndex] > connectionThreshold) return null;
  const root = stemEnds[stemElbowIndex === 0 ? 1 : 0];
  const branchVector = { x: elbow.x - tip.x, y: elbow.y - tip.y };
  const stemVector = { x: root.x - elbow.x, y: root.y - elbow.y };
  const branchLength = Math.hypot(branchVector.x, branchVector.y);
  const stemLength = Math.hypot(stemVector.x, stemVector.y);
  if (branchLength < boundsDiagonal(markerBounds) * 1.5 || stemLength < boundsDiagonal(markerBounds) * 1.5) return null;
  const bend = Math.abs(branchVector.x * stemVector.y - branchVector.y * stemVector.x) / Math.max(1e-9, branchLength * stemLength);
  if (bend < 0.25) return null;
  const wingProjection = (vector) => (vector.x * branchVector.x + vector.y * branchVector.y) / Math.max(1e-9, Math.hypot(vector.x, vector.y) * branchLength);
  if (wingProjection(firstWingVector) < 0.2 || wingProjection(secondWingVector) < 0.2) return null;
  return {
    authoredTextOp,
    pathOps: [branchIndex, firstWingIndex, secondWingIndex, stemIndex],
    markerOps: [firstWingIndex, secondWingIndex],
    endpoints: [root, tip],
    tip
  };
};
var joinedEndpoints = (left, right, threshold) => {
  const pairs = [
    { distance: pointDistance(left[0], right[0]), outer: [left[1], right[1]] },
    { distance: pointDistance(left[0], right[1]), outer: [left[1], right[0]] },
    { distance: pointDistance(left[1], right[0]), outer: [left[0], right[1]] },
    { distance: pointDistance(left[1], right[1]), outer: [left[0], right[0]] }
  ].sort((a, b) => a.distance - b.distance);
  return pairs[0].distance <= threshold ? pairs[0].outer : null;
};
var forwardPathChain = (scene, firstIndex, firstEndpoints, pageDiagonal) => {
  const pathOps = [firstIndex];
  let endpoints = firstEndpoints;
  for (let nextIndex = firstIndex + 1; nextIndex <= firstIndex + 7; nextIndex += 1) {
    const next = scene.ops[nextIndex];
    if (next?.kind !== "path" || !next.stroke || next.fill || next.segments.length < 2 || next.segments.length > 4) break;
    const nextEndpoints = pathEndpoints(next);
    const joined = nextEndpoints && joinedEndpoints(endpoints, nextEndpoints, pageDiagonal * 2e-3);
    if (!joined) break;
    endpoints = joined;
    pathOps.push(nextIndex);
  }
  return { pathOps, endpoints };
};
var splitArrowFanoutRootBefore = (scene, branchIndex, branchEndpoints, markerOps, segmentation, pageDiagonal) => {
  if (markerOps.length !== 2) return null;
  const markerBounds = unionBounds(
    finiteBounds(scene.ops[markerOps[0]].bounds),
    finiteBounds(scene.ops[markerOps[1]].bounds)
  );
  const markerDistances = branchEndpoints.map((endpoint) => pointToBounds(endpoint, markerBounds));
  const markerEndpointIndex = markerDistances[0] <= markerDistances[1] ? 0 : 1;
  const markerEndpoint = branchEndpoints[markerEndpointIndex];
  const junction = branchEndpoints[markerEndpointIndex === 0 ? 1 : 0];
  const connectionThreshold = pageDiagonal * 12e-4;
  if (markerDistances[markerEndpointIndex] > connectionThreshold || markerDistances[markerEndpointIndex === 0 ? 1 : 0] <= connectionThreshold * 2) return null;
  const group = groupForOp(segmentation, branchIndex);
  const followingText = scene.ops.slice(branchIndex + 1, branchIndex + 25).map((op, offset) => ({ op, opIndex: branchIndex + offset + 1 })).filter((entry) => entry.op.kind === "text" && entry.op.text.trim().length > 0 && groupForOp(segmentation, entry.opIndex) === group);
  if (!followingText.length) return null;
  const firstMarker = Math.min(...markerOps);
  for (let stubIndex = firstMarker - 1; stubIndex >= Math.max(0, firstMarker - 8); stubIndex -= 1) {
    const stub = scene.ops[stubIndex];
    if (groupForOp(segmentation, stubIndex) !== group || stub?.kind !== "path" || !stub.stroke || stub.fill || stub.segments.length < 2 || stub.segments.length > 4) continue;
    const stubEndpoints = pathEndpoints(stub);
    if (!stubEndpoints) continue;
    const junctionDistances = stubEndpoints.map((endpoint) => pointDistance(endpoint, junction));
    const junctionEndIndex = junctionDistances[0] <= junctionDistances[1] ? 0 : 1;
    if (junctionDistances[junctionEndIndex] > connectionThreshold) continue;
    const root = stubEndpoints[junctionEndIndex === 0 ? 1 : 0];
    const rootTextDistance = Math.min(...followingText.map(({ op }) => pointToBounds(root, finiteBounds(op.bounds))));
    const junctionTextDistance = Math.min(...followingText.map(({ op }) => pointToBounds(junction, finiteBounds(op.bounds))));
    if (rootTextDistance > pageDiagonal * 4e-3 || rootTextDistance + connectionThreshold >= junctionTextDistance) continue;
    return {
      stubIndex,
      endpoints: [root, markerEndpoint],
      ownsStub: stubIndex === firstMarker - 1
    };
  }
  return null;
};
var compoundOutlinedLeader = (scene, opIndex, pageDiagonal) => {
  const op = scene.ops[opIndex];
  const marker = scene.ops[opIndex - 1];
  if (op?.kind !== "path" || !op.stroke || op.fill || op.segments.length < 5 || op.segments.length > 12 || marker?.kind !== "path" || !marker.stroke || marker.fill) return null;
  const endpoints = pathEndpoints(op);
  if (!endpoints || boundsDiagonal(op.bounds) < pageDiagonal * 6e-3 || boundsDiagonal(marker.bounds) > pageDiagonal * 6e-3) return null;
  const vertices = pathVertices(op);
  const lengths = vertices.slice(1).map((point, index) => pointDistance(vertices[index], point)).sort((left, right) => right - left);
  if (!lengths[0] || lengths[0] < pageDiagonal * 6e-3 || lengths[0] < Math.max(pageDiagonal * 3e-3, (lengths[1] ?? 0) * 2.5)) return null;
  const markerDistance = Math.min(
    pointToPath(endpoints[0], marker),
    pointToPath(endpoints[1], marker)
  );
  if (markerDistance > pageDiagonal * 2e-3) return null;
  return { endpoints, markerOp: opIndex - 1 };
};
var leaderCandidates = (scene, pageDiagonal, segmentation, carrierMembers = /* @__PURE__ */ new Set()) => {
  const candidates = [];
  const consumed = /* @__PURE__ */ new Set();
  const openMarkerPackets = /* @__PURE__ */ new Map();
  const openMarkerMembers = /* @__PURE__ */ new Set();
  for (let opIndex = 2; opIndex < scene.ops.length; opIndex += 1) {
    const marker = openEndpointMarkerBefore(scene, opIndex, pageDiagonal);
    if (!marker || scene.ops[opIndex - 3]?.kind !== "text" && !carrierMembers.has(opIndex - 3)) continue;
    openMarkerPackets.set(opIndex, marker);
    marker.ops.forEach((index) => openMarkerMembers.add(index));
  }
  const foldedOpenPackets = /* @__PURE__ */ new Map();
  const foldedOpenMembers = /* @__PURE__ */ new Set();
  for (let opIndex = 4; opIndex < scene.ops.length; opIndex += 1) {
    const packet = foldedOpenEndpointPacketBefore(scene, opIndex, pageDiagonal, segmentation);
    if (!packet) continue;
    foldedOpenPackets.set(opIndex, packet);
    packet.pathOps.slice(0, -1).forEach((index) => foldedOpenMembers.add(index));
  }
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    if (consumed.has(opIndex) || openMarkerMembers.has(opIndex) || foldedOpenMembers.has(opIndex)) continue;
    const op = scene.ops[opIndex];
    const compoundOutlined = compoundOutlinedLeader(scene, opIndex, pageDiagonal);
    if (compoundOutlined) {
      candidates.push({
        opIndex,
        op,
        pathOps: [opIndex],
        endpoints: compoundOutlined.endpoints,
        arrowheadOps: [compoundOutlined.markerOp],
        packetKind: "arrow-leader",
        markerKind: "outline-marker",
        markerOps: [compoundOutlined.markerOp]
      });
      continue;
    }
    if (op.kind !== "path" || !op.stroke || op.fill || op.segments.length < 2 || op.segments.length > 4) continue;
    let endpoints = pathEndpoints(op);
    if (!endpoints) continue;
    const diagonal = boundsDiagonal(op.bounds);
    if (diagonal < pageDiagonal * 1e-3 || diagonal > pageDiagonal * 0.32) continue;
    const foldedOpen = foldedOpenPackets.get(opIndex);
    if (foldedOpen) {
      candidates.push({
        opIndex,
        op,
        pathOps: foldedOpen.pathOps,
        endpoints: foldedOpen.endpoints,
        arrowheadOps: [],
        packetKind: "open-marker-leader",
        markerKind: "open-marker",
        markerOps: foldedOpen.markerOps,
        authoredTextOp: foldedOpen.authoredTextOp,
        openMarkerTip: foldedOpen.tip
      });
      continue;
    }
    const openMarker = openMarkerPackets.get(opIndex);
    if (openMarker) {
      candidates.push({
        opIndex,
        op,
        pathOps: [opIndex, ...openMarker.ops],
        endpoints,
        arrowheadOps: [],
        packetKind: "open-marker-leader",
        markerKind: "open-marker",
        markerOps: [...openMarker.ops],
        authoredTextOp: opIndex - 3,
        openMarkerTip: openMarker.tip
      });
      continue;
    }
    const compoundMarker = compoundEndpointMarkerBefore(scene, opIndex, endpoints, pageDiagonal, segmentation);
    if (compoundMarker.length) {
      candidates.push({
        opIndex,
        op,
        pathOps: [opIndex],
        endpoints,
        arrowheadOps: compoundMarker,
        packetKind: "arrow-leader",
        markerKind: "compound-marker",
        markerOps: [...compoundMarker]
      });
      continue;
    }
    const compoundArrow = splitArrowBefore(scene, opIndex, pageDiagonal);
    if (compoundArrow.length) {
      const chain = forwardPathChain(scene, opIndex, endpoints, pageDiagonal);
      const fanoutRoot = splitArrowFanoutRootBefore(
        scene,
        opIndex,
        chain.endpoints,
        compoundArrow,
        segmentation,
        pageDiagonal
      );
      endpoints = fanoutRoot?.endpoints ?? chain.endpoints;
      const pathOps = fanoutRoot?.ownsStub ? [fanoutRoot.stubIndex, ...chain.pathOps] : chain.pathOps;
      if (markerBundleTouchesEndpoint(
        scene,
        compoundArrow,
        endpoints,
        pageDiagonal,
        Math.max(...pathOps.map((pathOp) => {
          const candidate = scene.ops[pathOp];
          return candidate?.kind === "path" ? candidate.lineWidth : 0;
        }))
      )) {
        pathOps.slice(1).forEach((index) => consumed.add(index));
        const packetIndex = pathOps.at(-1);
        candidates.push({ opIndex: packetIndex, op: scene.ops[packetIndex], pathOps, endpoints, arrowheadOps: compoundArrow, packetKind: "arrow-leader", markerKind: "split-arrow", markerOps: [...compoundArrow] });
        continue;
      }
    }
    const previous = scene.ops[opIndex - 1];
    const previousChain = isCompactArrowhead(previous, pageDiagonal) ? forwardPathChain(scene, opIndex, endpoints, pageDiagonal) : void 0;
    const previousMarkerAttached = isCompactArrowhead(previous, pageDiagonal) && markerTouchesEndpoint(
      previous,
      previousChain.endpoints,
      pageDiagonal,
      boundsDiagonal(previous.bounds),
      1,
      op.lineWidth
    );
    if (previousMarkerAttached) {
      const chain = previousChain;
      chain.pathOps.slice(1).forEach((index) => consumed.add(index));
      const packetIndex = chain.pathOps.at(-1);
      candidates.push({
        opIndex: packetIndex,
        op: scene.ops[packetIndex],
        pathOps: chain.pathOps,
        endpoints: chain.endpoints,
        arrowheadOps: [opIndex - 1],
        packetKind: "arrow-leader",
        markerKind: "filled-arrow",
        markerOps: [opIndex - 1]
      });
      continue;
    }
    const interleavedArrow = scene.ops[opIndex + 1];
    const interleavedFollowing = scene.ops[opIndex + 2];
    const interleavedFollowingEndpoints = isCompactArrowhead(interleavedArrow, pageDiagonal) && interleavedFollowing?.kind === "path" && interleavedFollowing.stroke && !interleavedFollowing.fill && interleavedFollowing.segments.length >= 2 && interleavedFollowing.segments.length <= 4 ? pathEndpoints(interleavedFollowing) : null;
    const interleavedJoined = interleavedFollowingEndpoints && joinedEndpoints(endpoints, interleavedFollowingEndpoints, pageDiagonal * 2e-3);
    const interleavedMarkerAttached = interleavedJoined && isCompactArrowhead(interleavedArrow, pageDiagonal) && markerTouchesEndpoint(
      interleavedArrow,
      interleavedJoined,
      pageDiagonal,
      boundsDiagonal(interleavedArrow.bounds),
      1,
      Math.max(op.lineWidth, interleavedFollowing.lineWidth)
    );
    if (interleavedJoined && interleavedMarkerAttached) {
      consumed.add(opIndex + 2);
      candidates.push({
        opIndex: opIndex + 2,
        op: interleavedFollowing,
        pathOps: [opIndex, opIndex + 2],
        endpoints: interleavedJoined,
        arrowheadOps: [opIndex + 1],
        packetKind: "arrow-leader",
        markerKind: "filled-arrow",
        markerOps: [opIndex + 1]
      });
      continue;
    }
    const next = scene.ops[opIndex + 1];
    if (isCompactArrowhead(next, pageDiagonal)) {
      const following = scene.ops[opIndex + 2];
      const followingEndpoints = following?.kind === "path" && following.stroke && !following.fill && following.segments.length >= 2 && following.segments.length <= 4 ? pathEndpoints(following) : null;
      const joined = followingEndpoints && joinedEndpoints(endpoints, followingEndpoints, pageDiagonal * 2e-3);
      const markerAttached = markerTouchesEndpoint(
        next,
        joined ?? endpoints,
        pageDiagonal,
        boundsDiagonal(next.bounds),
        1,
        Math.max(op.lineWidth, following?.kind === "path" ? following.lineWidth : 0)
      );
      if (joined && markerAttached) {
        consumed.add(opIndex + 2);
        candidates.push({
          opIndex: opIndex + 2,
          op: following,
          pathOps: [opIndex, opIndex + 2],
          endpoints: joined,
          arrowheadOps: [opIndex + 1],
          packetKind: "arrow-leader",
          markerKind: "filled-arrow",
          markerOps: [opIndex + 1]
        });
        continue;
      }
      if (markerAttached) {
        candidates.push({ opIndex, op, pathOps: [opIndex], endpoints, arrowheadOps: [opIndex + 1], packetKind: "arrow-leader", markerKind: "filled-arrow", markerOps: [opIndex + 1] });
        continue;
      }
    }
    const framedCandidate = scene.ops[opIndex + 1];
    const framedPath = framedCandidate?.kind === "path" ? framedCandidate : void 0;
    const framedTextFollows = Boolean(framedPath?.fill && framedPath.stroke && framedPath.segments.length === 9 && scene.ops[opIndex + 2]?.kind === "text");
    const authoredTextOp = previous?.kind === "text" ? opIndex - 1 : scene.ops[opIndex - 2]?.kind === "text" ? opIndex - 2 : framedTextFollows ? opIndex + 2 : void 0;
    if (authoredTextOp !== void 0) candidates.push({
      opIndex,
      op,
      pathOps: [opIndex],
      endpoints,
      arrowheadOps: [],
      packetKind: "leader-only",
      markerKind: "none",
      markerOps: [],
      authoredTextOp
    });
  }
  return candidates;
};
var isOutlineGlyphPath = (op, pageDiagonal) => Boolean(op?.kind === "path" && op.stroke && !op.fill && op.segments.length >= 2 && op.segments.length <= 40 && boundsDiagonal(op.bounds) >= pageDiagonal * 15e-5 && boundsDiagonal(op.bounds) <= pageDiagonal * 45e-4);
var median = (values) => {
  const sorted = [...values].sort((left, right) => left - right);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
var vectorSymbolClusters = (scene, segmentation, pageDiagonal) => {
  if (!segmentation) return [];
  const symbols = [];
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    const enclosure = scene.ops[opIndex];
    if (enclosure.kind !== "path" || !enclosure.stroke || enclosure.fill || enclosure.segments.length !== 7 || opIndex < 6) continue;
    const enclosureBounds = finiteBounds(enclosure.bounds);
    const enclosureDiagonal = boundsDiagonal(enclosureBounds);
    const width = boundsWidth(enclosureBounds);
    const height = boundsHeight(enclosureBounds);
    if (enclosureDiagonal < pageDiagonal * 4e-3 || enclosureDiagonal > pageDiagonal * 0.018 || width < height * 0.7 || height < width * 0.7) continue;
    const endpoints = pathEndpoints(enclosure);
    if (!endpoints || pointDistance(endpoints[0], endpoints[1]) > Math.max(
      pageDiagonal * 2e-4,
      enclosureDiagonal * 0.08
    )) continue;
    const group = groupForOp(segmentation, opIndex);
    const decodedTag = scene.ops[opIndex + 1];
    if (decodedTag?.kind === "text" && groupForOp(segmentation, opIndex + 1) === group) {
      const normalizedTag = decodedTag.text.replace(/\s+/g, "").trim().toUpperCase();
      const tagBounds = finiteBounds(decodedTag.bounds);
      const tagFits = /^[A-Z0-9]{1,3}$/.test(normalizedTag) && boundsDiagonal(tagBounds) <= enclosureDiagonal * 0.72 && boundsOverlapFraction(tagBounds, enclosureBounds) >= 0.78;
      if (tagFits) {
        let earlierFrameIndex = -1;
        for (let candidateIndex = opIndex - 1; candidateIndex >= 0; candidateIndex -= 1) {
          const candidate = scene.ops[candidateIndex];
          if (candidate.kind !== "path" || !candidate.stroke || candidate.fill || candidate.segments.length !== 7) continue;
          const candidateBounds = finiteBounds(candidate.bounds);
          if (boundsOverlapFraction(candidateBounds, enclosureBounds) < 0.98 || boundsOverlapFraction(enclosureBounds, candidateBounds) < 0.98) continue;
          const earlierMarker = scene.ops[candidateIndex + 1];
          const earlierLeader = scene.ops[candidateIndex + 2];
          const frameGroup = groupForOp(segmentation, candidateIndex);
          if (!isCompactArrowhead(earlierMarker, pageDiagonal) || earlierLeader?.kind !== "path" || !earlierLeader.stroke || earlierLeader.fill || earlierLeader.segments.length < 2 || earlierLeader.segments.length > 4 || groupForOp(segmentation, candidateIndex + 1) !== frameGroup || groupForOp(segmentation, candidateIndex + 2) !== frameGroup) continue;
          const leaderEndpoints = pathEndpoints(earlierLeader);
          if (!leaderEndpoints) continue;
          const markerThreshold = Math.max(
            pageDiagonal * 35e-5,
            boundsDiagonal(earlierMarker.bounds) * 0.25
          );
          const frameThreshold = Math.max(
            pageDiagonal * 35e-5,
            enclosureDiagonal * 0.08
          );
          const frameDistances = leaderEndpoints.map((endpoint) => pointToPath(endpoint, candidate));
          const markerDistances = leaderEndpoints.map((endpoint) => pointToBounds(endpoint, finiteBounds(earlierMarker.bounds)));
          const oppositeEndsAttached = frameDistances[0] <= frameThreshold && markerDistances[1] <= markerThreshold || frameDistances[1] <= frameThreshold && markerDistances[0] <= markerThreshold;
          if (!oppositeEndsAttached) continue;
          earlierFrameIndex = candidateIndex;
          break;
        }
        if (earlierFrameIndex < 0) continue;
        symbols.push({
          opIndices: [earlierFrameIndex, opIndex, opIndex + 1],
          ops: [decodedTag],
          bounds: unionBounds(enclosureBounds, tagBounds),
          text: decodedTag.text.trim(),
          scale: Math.max(1, textScale(decodedTag)),
          source: "vector-outline",
          carrierKind: "symbol",
          symbolOpIndex: earlierFrameIndex,
          carrierBounds: enclosureBounds
        });
        continue;
      }
    }
    const enclosureOps = Array.from({ length: 7 }, (_, offset) => opIndex - 6 + offset);
    if (enclosureOps.some((index) => groupForOp(segmentation, index) !== group)) continue;
    const expectedChannels = ["fill", "stroke", "stroke", "fill", "stroke", "stroke"];
    const framePacket = enclosureOps.slice(0, 6).map((index) => scene.ops[index]);
    if (framePacket.some((candidate, index) => candidate.kind !== "path" || (expectedChannels[index] === "fill" ? !candidate.fill || candidate.stroke : !candidate.stroke || candidate.fill) || boundsOverlapFraction(finiteBounds(candidate.bounds), enclosureBounds) < 0.72)) continue;
    const digitOps = [];
    for (let candidateIndex = opIndex + 1; candidateIndex <= Math.min(scene.ops.length - 1, opIndex + 4); candidateIndex += 1) {
      if (groupForOp(segmentation, candidateIndex) !== group) break;
      const candidate = scene.ops[candidateIndex];
      if (!isOutlineGlyphPath(candidate, pageDiagonal) || boundsDiagonal(candidate.bounds) > enclosureDiagonal * 0.62 || boundsOverlapFraction(finiteBounds(candidate.bounds), enclosureBounds) < 0.78) break;
      digitOps.push(candidateIndex);
    }
    if (digitOps.length < 1 || digitOps.length > 3) continue;
    const opIndices = [.../* @__PURE__ */ new Set([...enclosureOps, ...digitOps])].sort((left, right) => left - right);
    const symbolBounds = opIndices.slice(1).reduce(
      (combined, index) => unionBounds(combined, finiteBounds(scene.ops[index].bounds)),
      enclosureBounds
    );
    const glyphSizes = digitOps.map((index) => boundsDiagonal(scene.ops[index].bounds));
    symbols.push({
      opIndices,
      ops: [],
      bounds: symbolBounds,
      text: `\u77E2\u91CF\u7B26\u53F7 #${group}:${opIndex}`,
      scale: Math.max(1, median(glyphSizes)),
      source: "vector-outline",
      carrierKind: "symbol",
      symbolOpIndex: opIndex
    });
  }
  return symbols.sort((left, right) => left.opIndices[0] - right.opIndices[0]);
};
var vectorSymbolClustersInRois = (scene, segmentation, pageDiagonal, rois) => {
  if (!segmentation || !rois.length) return [];
  const frames = [];
  const selectedByRoi = (bounds) => rois.some((roi) => {
    const target = finiteBounds(roi);
    if (boundsOverlapFraction(bounds, target) >= 0.5 || boundsOverlapFraction(target, bounds) >= 0.5) return true;
    const horizontalGap = Math.max(0, bounds.minX - target.maxX, target.minX - bounds.maxX);
    const verticalGap = Math.max(0, bounds.minY - target.maxY, target.minY - bounds.maxY);
    const horizontalOverlap = Math.max(0, Math.min(bounds.maxX, target.maxX) - Math.max(bounds.minX, target.minX));
    const verticalOverlap = Math.max(0, Math.min(bounds.maxY, target.maxY) - Math.max(bounds.minY, target.minY));
    const adjacencyThreshold = pageDiagonal * 12e-4;
    return horizontalGap <= adjacencyThreshold && verticalOverlap >= Math.min(boundsHeight(bounds), boundsHeight(target)) * 0.55 || verticalGap <= adjacencyThreshold && horizontalOverlap >= Math.min(boundsWidth(bounds), boundsWidth(target)) * 0.55;
  });
  const validFrameBounds = (bounds) => {
    const width = boundsWidth(bounds);
    const height = boundsHeight(bounds);
    const diagonal = boundsDiagonal(bounds);
    return diagonal >= pageDiagonal * 4e-3 && diagonal <= pageDiagonal * 0.018 && width >= height * 0.65 && height >= width * 0.65 && selectedByRoi(bounds);
  };
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    const op = scene.ops[opIndex];
    if (op.kind !== "path" || !op.stroke || op.fill) continue;
    const lineCount = op.segments.filter((segment) => segment.kind === "line").length;
    const curveCount = op.segments.filter((segment) => segment.kind === "curve").length;
    if (curveCount || lineCount < 3 || op.segments.length < 5 || op.segments.length > 10) continue;
    const bounds = finiteBounds(op.bounds);
    if (!validFrameBounds(bounds)) continue;
    const endpoints = pathEndpoints(op);
    if (!endpoints || pointDistance(endpoints[0], endpoints[1]) > Math.max(
      pageDiagonal * 2e-4,
      boundsDiagonal(bounds) * 0.08
    )) continue;
    const group = groupForOp(segmentation, opIndex);
    const layered = scene.ops.flatMap((candidate, candidateIndex) => {
      if (candidateIndex < Math.max(0, opIndex - 4) || candidateIndex > opIndex || groupForOp(segmentation, candidateIndex) !== group || candidate.kind !== "path" || !candidate.stroke || candidate.fill) return [];
      const candidateBounds = finiteBounds(candidate.bounds);
      const similarlySized = boundsDiagonal(candidateBounds) >= boundsDiagonal(bounds) * 0.8 && boundsDiagonal(candidateBounds) <= boundsDiagonal(bounds) * 1.2;
      return similarlySized && boundsOverlapFraction(candidateBounds, bounds) >= 0.72 && boundsOverlapFraction(bounds, candidateBounds) >= 0.72 ? [candidateIndex] : [];
    });
    frames.push({
      opIndices: [.../* @__PURE__ */ new Set([...layered, opIndex])].sort((left, right) => left - right),
      representativeOpIndex: opIndex,
      bounds
    });
  }
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    const op = scene.ops[opIndex];
    if (op.kind !== "path" || !op.stroke || op.fill || op.segments.length > 40) continue;
    const lineCount = op.segments.filter((segment) => segment.kind === "line").length;
    const curveCount = op.segments.filter((segment) => segment.kind === "curve").length;
    if (curveCount < 2 && lineCount < 8) continue;
    const bounds = finiteBounds(op.bounds);
    if (!validFrameBounds(bounds)) continue;
    const endpoints = pathEndpoints(op);
    if (!endpoints || pointDistance(endpoints[0], endpoints[1]) > Math.max(
      pageDiagonal * 2e-4,
      boundsDiagonal(bounds) * 0.08
    )) continue;
    frames.push({ opIndices: [opIndex], representativeOpIndex: opIndex, bounds });
  }
  for (let opIndex = 0; opIndex < scene.ops.length - 1; opIndex += 1) {
    const first = scene.ops[opIndex];
    const second = scene.ops[opIndex + 1];
    if (first.kind !== "path" || second.kind !== "path" || !first.stroke || first.fill || !second.stroke || second.fill || first.segments.length !== 3 || second.segments.length !== 3 || first.segments.filter((segment) => segment.kind === "curve").length !== 2 || second.segments.filter((segment) => segment.kind === "curve").length !== 2 || groupForOp(segmentation, opIndex) !== groupForOp(segmentation, opIndex + 1)) continue;
    const firstBounds = finiteBounds(first.bounds);
    const secondBounds = finiteBounds(second.bounds);
    const bounds = unionBounds(firstBounds, secondBounds);
    if (!validFrameBounds(bounds) || boundsWidth(firstBounds) < boundsWidth(bounds) * 0.8 || boundsWidth(secondBounds) < boundsWidth(bounds) * 0.8 || boundsHeight(firstBounds) > boundsHeight(bounds) * 0.68 || boundsHeight(secondBounds) > boundsHeight(bounds) * 0.68) continue;
    const firstEndpoints = pathEndpoints(first);
    const secondEndpoints = pathEndpoints(second);
    if (!firstEndpoints || !secondEndpoints) continue;
    const direct = pointDistance(firstEndpoints[0], secondEndpoints[0]) + pointDistance(firstEndpoints[1], secondEndpoints[1]);
    const reversed = pointDistance(firstEndpoints[0], secondEndpoints[1]) + pointDistance(firstEndpoints[1], secondEndpoints[0]);
    if (Math.min(direct, reversed) > boundsDiagonal(bounds) * 0.16) continue;
    frames.push({ opIndices: [opIndex, opIndex + 1], representativeOpIndex: opIndex, bounds });
  }
  const symbols = [];
  const uniqueFrames = frames.filter((frame, index) => frames.findIndex((candidate) => candidate.representativeOpIndex === frame.representativeOpIndex && candidate.opIndices.join(",") === frame.opIndices.join(",")) === index);
  for (const frame of uniqueFrames) {
    const group = groupForOp(segmentation, frame.representativeOpIndex);
    const lastFrame = Math.max(...frame.opIndices);
    const glyphOps = [];
    for (let candidateIndex = lastFrame + 1; candidateIndex <= Math.min(scene.ops.length - 1, lastFrame + 4); candidateIndex += 1) {
      if (groupForOp(segmentation, candidateIndex) !== group) break;
      const candidate = scene.ops[candidateIndex];
      if (!isOutlineGlyphPath(candidate, pageDiagonal) || boundsDiagonal(candidate.bounds) > boundsDiagonal(frame.bounds) * 0.65 || boundsOverlapFraction(finiteBounds(candidate.bounds), frame.bounds) < 0.78) break;
      glyphOps.push(candidateIndex);
    }
    if (glyphOps.length < 1 || glyphOps.length > 3) continue;
    const opIndices = [.../* @__PURE__ */ new Set([...frame.opIndices, ...glyphOps])].sort((left, right) => left - right);
    const symbolBounds = opIndices.slice(1).reduce(
      (combined, index) => unionBounds(combined, finiteBounds(scene.ops[index].bounds)),
      frame.bounds
    );
    symbols.push({
      opIndices,
      ops: [],
      bounds: symbolBounds,
      text: `ROI \u77E2\u91CF\u7B26\u53F7 #${group}:${frame.representativeOpIndex}`,
      scale: Math.max(1, median(glyphOps.map((index) => boundsDiagonal(scene.ops[index].bounds)))),
      source: "vector-outline",
      roiSeeded: true,
      carrierKind: "symbol",
      symbolOpIndex: frame.representativeOpIndex,
      carrierBounds: frame.bounds
    });
  }
  return symbols.sort((left, right) => left.opIndices[0] - right.opIndices[0]);
};
var mergeOutlineTextWithSymbols = (outlineText, symbols, segmentation, pageDiagonal) => {
  const consumedText = /* @__PURE__ */ new Set();
  const mergedSymbols = symbols.map((symbol) => {
    const symbolGroup = groupForOp(segmentation, symbol.opIndices[0]);
    const candidates = outlineText.filter((cluster) => {
      if (consumedText.has(cluster)) return false;
      const groupGap = Math.abs(groupForOp(segmentation, cluster.opIndices[0]) - symbolGroup);
      if (groupGap > 3 && !(symbol.roiSeeded && cluster.roiSeeded)) return false;
      const verticalOverlap = Math.max(
        0,
        Math.min(cluster.bounds.maxY, symbol.bounds.maxY) - Math.max(cluster.bounds.minY, symbol.bounds.minY)
      );
      const horizontalOverlap = Math.max(
        0,
        Math.min(cluster.bounds.maxX, symbol.bounds.maxX) - Math.max(cluster.bounds.minX, symbol.bounds.minX)
      );
      const aligned = verticalOverlap >= Math.min(boundsHeight(cluster.bounds), boundsHeight(symbol.bounds)) * 0.3 || horizontalOverlap >= Math.min(boundsWidth(cluster.bounds), boundsWidth(symbol.bounds)) * 0.3;
      return aligned && boundsGap(cluster.bounds, symbol.bounds) <= Math.max(
        pageDiagonal * 2e-3,
        Math.max(cluster.scale, symbol.scale) * 1.2
      );
    }).sort((left, right) => boundsGap(left.bounds, symbol.bounds) - boundsGap(right.bounds, symbol.bounds));
    const label = candidates[0];
    if (!label) return symbol;
    consumedText.add(label);
    const opIndices = [.../* @__PURE__ */ new Set([...label.opIndices, ...symbol.opIndices])].sort((left, right) => left - right);
    return {
      ...symbol,
      opIndices,
      bounds: unionBounds(label.bounds, symbol.bounds),
      text: `${label.text} + ${symbol.text}`,
      scale: Math.max(label.scale, symbol.scale),
      carrierKind: "text-symbol"
    };
  });
  return [
    ...outlineText.filter((cluster) => !consumedText.has(cluster)),
    ...mergedSymbols
  ].sort((left, right) => left.opIndices[0] - right.opIndices[0]);
};
var vectorOutlineTextClusters = (scene, segmentation, pageDiagonal) => {
  if (!segmentation) return [];
  const rawClusters = [];
  const maxRunLength = 220;
  const maxSequentialJump = pageDiagonal * 6e-3;
  const acceptRun = (indices) => {
    if (indices.length < 8) return;
    const ops = indices.map((index) => scene.ops[index]).filter((op) => isOutlineGlyphPath(op, pageDiagonal));
    if (ops.length !== indices.length) return;
    const nontrivial = ops.filter((op) => op.segments.length >= 3).length;
    const simpleRatio = ops.filter((op) => op.segments.length === 2).length / ops.length;
    if (nontrivial < 2 || nontrivial / ops.length < 0.18 || simpleRatio < 0.2 || simpleRatio > 0.82) return;
    const sizes = ops.map((op) => Math.max(boundsWidth(op.bounds), boundsHeight(op.bounds)));
    const scale = Math.max(1, median(sizes));
    const sortedSizes = [...sizes].sort((left, right) => left - right);
    const q90 = sortedSizes[Math.min(sortedSizes.length - 1, Math.floor(sortedSizes.length * 0.9))];
    if (q90 > scale * 1.8) return;
    const bounds = ops.reduce((combined, op) => unionBounds(combined, finiteBounds(op.bounds)), finiteBounds(ops[0].bounds));
    const majorSpan = Math.max(boundsWidth(bounds), boundsHeight(bounds));
    const minorSpan = Math.min(boundsWidth(bounds), boundsHeight(bounds));
    if (majorSpan < scale * 1.8 || boundsDiagonal(bounds) > pageDiagonal * 0.16 || minorSpan > pageDiagonal * 0.035) return;
    const first = indices[0];
    const group = groupForOp(segmentation, first);
    rawClusters.push({
      opIndices: indices,
      ops: [],
      bounds,
      text: `\u77E2\u91CF\u8F6E\u5ED3\u6587\u5B57 #${group}:${first}`,
      scale,
      source: "vector-outline"
    });
  };
  let run = [];
  const flush = () => {
    acceptRun(run);
    run = [];
  };
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    const op = scene.ops[opIndex];
    if (!isOutlineGlyphPath(op, pageDiagonal)) {
      flush();
      continue;
    }
    const previousIndex = run.at(-1);
    const continues = previousIndex !== void 0 && run.length < maxRunLength && groupForOp(segmentation, previousIndex) === groupForOp(segmentation, opIndex) && boundsGap(finiteBounds(scene.ops[previousIndex].bounds), finiteBounds(op.bounds)) <= maxSequentialJump;
    if (!continues) flush();
    run.push(opIndex);
  }
  flush();
  const merged = [];
  for (const cluster of rawClusters) {
    const group = groupForOp(segmentation, cluster.opIndices[0]);
    const candidates = merged.filter((item) => {
      const itemGroup = groupForOp(segmentation, item.opIndices[0]);
      const scale = Math.max(item.scale, cluster.scale);
      if (Math.abs(itemGroup - group) > 2 || scale / Math.max(1, Math.min(item.scale, cluster.scale)) > 2) return false;
      const itemHorizontal = boundsWidth(item.bounds) >= boundsHeight(item.bounds) * 1.5;
      const clusterHorizontal = boundsWidth(cluster.bounds) >= boundsHeight(cluster.bounds) * 1.5;
      if (itemHorizontal !== clusterHorizontal) return false;
      const lineGap = itemHorizontal ? Math.max(0, item.bounds.minY - cluster.bounds.maxY, cluster.bounds.minY - item.bounds.maxY) : Math.max(0, item.bounds.minX - cluster.bounds.maxX, cluster.bounds.minX - item.bounds.maxX);
      const aligned = itemHorizontal ? Math.abs(item.bounds.minX - cluster.bounds.minX) : Math.abs(item.bounds.minY - cluster.bounds.minY);
      return lineGap <= Math.max(pageDiagonal * 12e-4, scale * 0.8) && aligned <= Math.max(pageDiagonal * 25e-4, scale * 2.2);
    }).sort((left, right) => boundsGap(left.bounds, cluster.bounds) - boundsGap(right.bounds, cluster.bounds));
    const target = candidates[0];
    if (target && boundsDiagonal(unionBounds(target.bounds, cluster.bounds)) <= pageDiagonal * 0.16) {
      target.opIndices = [...target.opIndices, ...cluster.opIndices].sort((left, right) => left - right);
      target.bounds = unionBounds(target.bounds, cluster.bounds);
      target.scale = Math.max(target.scale, cluster.scale);
    } else {
      merged.push({ ...cluster, opIndices: [...cluster.opIndices] });
    }
  }
  for (const cluster of merged) {
    const group = groupForOp(segmentation, cluster.opIndices[0]);
    cluster.text = `\u77E2\u91CF\u8F6E\u5ED3\u6587\u5B57 #${group}:${cluster.opIndices[0]}`;
  }
  return merged.filter((cluster) => {
    const minorSpan = Math.min(boundsWidth(cluster.bounds), boundsHeight(cluster.bounds));
    return cluster.opIndices.length >= 20 && minorSpan <= pageDiagonal * 0.035;
  }).sort((left, right) => left.opIndices[0] - right.opIndices[0]);
};
var relaxedVectorOutlineTextRuns = (scene, segmentation, pageDiagonal) => {
  if (!segmentation) return [];
  const clusters = [];
  let run = [];
  const flush = () => {
    const indices = run;
    run = [];
    if (indices.length < 7 || indices.length > 260) return;
    const ops = indices.map((index) => scene.ops[index]).filter((op) => isOutlineGlyphPath(op, pageDiagonal));
    if (ops.length !== indices.length) return;
    if (ops.filter((op) => op.segments.length >= 3).length < 1) return;
    const sizes = ops.map((op) => Math.max(boundsWidth(op.bounds), boundsHeight(op.bounds)));
    const scale = Math.max(1, median(sizes));
    const sortedSizes = [...sizes].sort((left, right) => left - right);
    const q90 = sortedSizes[Math.min(sortedSizes.length - 1, Math.floor(sortedSizes.length * 0.9))];
    if (q90 > scale * 1.8) return;
    const bounds = ops.slice(1).reduce(
      (combined, op) => unionBounds(combined, finiteBounds(op.bounds)),
      finiteBounds(ops[0].bounds)
    );
    const majorSpan = Math.max(boundsWidth(bounds), boundsHeight(bounds));
    const minorSpan = Math.min(boundsWidth(bounds), boundsHeight(bounds));
    if (majorSpan < scale * 1.8 || boundsDiagonal(bounds) > pageDiagonal * 0.16 || minorSpan > pageDiagonal * 0.035) return;
    const group = groupForOp(segmentation, indices[0]);
    clusters.push({
      opIndices: indices,
      ops: [],
      bounds,
      text: `\u77E2\u91CF\u8F6E\u5ED3\u6587\u5B57 #${group}:${indices[0]}`,
      scale,
      source: "vector-outline",
      roiSeeded: true
    });
  };
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    const op = scene.ops[opIndex];
    const previousIndex = run.at(-1);
    const previousOp = previousIndex === void 0 ? void 0 : scene.ops[previousIndex];
    const continues = isOutlineGlyphPath(op, pageDiagonal) && (previousIndex === void 0 || opIndex === previousIndex + 1 && groupForOp(segmentation, previousIndex) === groupForOp(segmentation, opIndex) && previousOp?.kind === "path" && boundsGap(finiteBounds(previousOp.bounds), finiteBounds(op.bounds)) <= pageDiagonal * 6e-3);
    if (!continues) flush();
    if (isOutlineGlyphPath(op, pageDiagonal)) run.push(opIndex);
  }
  flush();
  return clusters;
};
var vectorOutlineTextClustersInRoi = (scene, pageDiagonal, roi, relaxedRuns) => {
  const target = finiteBounds(roi);
  return relaxedRuns.filter((cluster) => {
    const selectedCount = cluster.opIndices.filter((index) => intersects2(finiteBounds(scene.ops[index].bounds), target)).length;
    const selectedRatio = selectedCount / Math.max(1, cluster.opIndices.length);
    const sharedFraction = Math.max(
      boundsOverlapFraction(cluster.bounds, target),
      boundsOverlapFraction(target, cluster.bounds)
    );
    const clusterMajor = Math.max(boundsWidth(cluster.bounds), boundsHeight(cluster.bounds));
    const targetMajor = Math.max(boundsWidth(target), boundsHeight(target));
    return selectedCount >= 7 && selectedRatio >= 0.2 && sharedFraction >= 0.15 && clusterMajor <= Math.max(pageDiagonal * 6e-3, targetMajor * 5) && boundsDiagonal(cluster.bounds) <= pageDiagonal * 0.16;
  });
};
var hasMarkersAtBothEnds = (scene, opIndex, endpoints, pageDiagonal) => {
  const nearEndpoint = [false, false];
  const from = Math.max(0, opIndex - 6);
  const to = Math.min(scene.ops.length - 1, opIndex + 6);
  for (let candidateIndex = from; candidateIndex <= to; candidateIndex += 1) {
    if (candidateIndex === opIndex) continue;
    const candidate = scene.ops[candidateIndex];
    if (candidate.kind !== "path" || !candidate.fill || candidate.segments.length < 4 || candidate.segments.length > 12) continue;
    const diagonal = boundsDiagonal(candidate.bounds);
    if (diagonal < pageDiagonal * 25e-5 || diagonal > pageDiagonal * 0.012) continue;
    endpoints.forEach((endpoint, endpointIndex) => {
      if (pointToBounds(endpoint, finiteBounds(candidate.bounds)) <= pageDiagonal * 3e-3) nearEndpoint[endpointIndex] = true;
    });
  }
  return nearEndpoint[0] && nearEndpoint[1];
};
var outlineLeaderCandidates = (scene, outlineClusters, existing, pageDiagonal) => {
  if (!outlineClusters.length) return [];
  const textMembers = new Set(outlineClusters.flatMap((cluster) => cluster.opIndices));
  const existingPaths = new Set(existing.flatMap((leader) => leader.pathOps));
  const candidates = [];
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    if (textMembers.has(opIndex) || existingPaths.has(opIndex)) continue;
    const op = scene.ops[opIndex];
    if (op.kind !== "path" || !op.stroke || op.fill || op.segments.length < 2 || op.segments.length > 4) continue;
    const endpoints = pathEndpoints(op);
    if (!endpoints) continue;
    if (hasMarkersAtBothEnds(scene, opIndex, endpoints, pageDiagonal)) continue;
    const diagonal = boundsDiagonal(op.bounds);
    if (diagonal < pageDiagonal * 1e-3 || diagonal > pageDiagonal * 0.32) continue;
    const nearOutlineText = outlineClusters.some((cluster) => {
      const firstDistance = pointToBounds(endpoints[0], cluster.bounds);
      const secondDistance = pointToBounds(endpoints[1], cluster.bounds);
      const rootDistance = Math.min(firstDistance, secondDistance);
      const targetDistance = Math.max(firstDistance, secondDistance);
      const threshold = Math.min(
        pageDiagonal * 0.018,
        Math.max(cluster.scale * DEFAULT_CALLOUT_DETECTION_OPTIONS.leaderRootDistanceScale, pageDiagonal * 25e-4)
      );
      return rootDistance <= threshold && targetDistance >= threshold * 0.65;
    });
    if (nearOutlineText) {
      candidates.push({ opIndex, op, pathOps: [opIndex], endpoints, arrowheadOps: [], packetKind: "leader-only", markerKind: "none", markerOps: [] });
    }
  }
  return candidates;
};
var arrowheadCandidates = (scene, pageDiagonal) => scene.ops.flatMap((op, opIndex) => isCompactArrowhead(op, pageDiagonal) ? [{ opIndex, op }] : []);
var expandOutlineLeader = (scene, leader, cluster, arrowheads, outlineMembers, segmentation, pageDiagonal, options) => {
  if (cluster.source !== "vector-outline") return leader;
  const [first, second] = leader.endpoints;
  const firstDistance = pointToBounds(first, cluster.bounds);
  const root = firstDistance <= pointToBounds(second, cluster.bounds) ? first : second;
  const target = root === first ? second : first;
  const markerThreshold = Math.max(cluster.scale * 2.5, pageDiagonal * 4e-3);
  const leaderGroup = groupForOp(segmentation, leader.opIndex);
  const allowCrossBatchTopology = Boolean(cluster.roiSeeded && leader.packetKind === "leader-only");
  const spatialIndex = allowCrossBatchTopology ? getSceneSpatialIndex(scene) : void 0;
  const spatialSearchRadius = Math.min(
    pageDiagonal * 3e-3,
    Math.max(
      pageDiagonal * 8e-4,
      cluster.scale * options.leaderConnectionScale * 0.5,
      leader.op.lineWidth * 4
    )
  );
  const markerPool = /* @__PURE__ */ new Map();
  for (const candidate of arrowheads) {
    if (Math.abs(candidate.opIndex - leader.opIndex) > 400 || Math.abs(groupForOp(segmentation, candidate.opIndex) - leaderGroup) > 24) continue;
    markerPool.set(candidate.opIndex, candidate.op);
  }
  for (const opIndex of leader.arrowheadOps) {
    const op = scene.ops[opIndex];
    if (op?.kind === "path") markerPool.set(opIndex, op);
  }
  const localMarkerOps = new Set(markerPool.keys());
  const nearestMarker = (point) => {
    const pool = new Map(markerPool);
    if (spatialIndex) {
      for (const opIndex of spatialIndex.queryPoint(point.x, point.y, spatialSearchRadius)) {
        const op = scene.ops[opIndex];
        if (!isCompactArrowhead(op, pageDiagonal) || Math.abs(groupForOp(segmentation, opIndex) - leaderGroup) > 24) continue;
        pool.set(opIndex, op);
      }
    }
    const ranked = [...pool].flatMap(([opIndex, op]) => {
      const distance = allowCrossBatchTopology ? pointToPath(point, op) : pointToBounds(point, finiteBounds(op.bounds));
      const threshold = allowCrossBatchTopology ? endpointContactThreshold(
        pageDiagonal,
        cluster.scale,
        options.leaderConnectionScale,
        Math.max(leader.op.lineWidth, op.lineWidth),
        boundsDiagonal(op.bounds)
      ) : markerThreshold;
      return distance <= threshold ? [{
        opIndex,
        distance,
        local: localMarkerOps.has(opIndex)
      }] : [];
    }).sort((left, right) => {
      const distanceDelta = left.distance - right.distance;
      const equivalentDistance = Math.max(pageDiagonal * 1e-5, spatialSearchRadius * 0.05);
      if (Math.abs(distanceDelta) > equivalentDistance) return distanceDelta;
      if (left.local !== right.local) return left.local ? -1 : 1;
      const leftGroupGap = Math.abs(groupForOp(segmentation, left.opIndex) - leaderGroup);
      const rightGroupGap = Math.abs(groupForOp(segmentation, right.opIndex) - leaderGroup);
      return leftGroupGap - rightGroupGap || Math.abs(left.opIndex - leader.opIndex) - Math.abs(right.opIndex - leader.opIndex) || left.opIndex - right.opIndex;
    });
    if (allowCrossBatchTopology && ranked.length !== 1) return void 0;
    return ranked[0];
  };
  const currentDx = target.x - root.x;
  const currentDy = target.y - root.y;
  const currentLength = Math.max(1e-9, Math.hypot(currentDx, currentDy));
  const directMarkerCandidate = nearestMarker(target);
  const directMarker = directMarkerCandidate && (!allowCrossBatchTopology || spatialTriangleMarkerMatches(
    scene,
    directMarkerCandidate.opIndex,
    leader.pathOps,
    root,
    target,
    currentLength,
    pageDiagonal
  )) ? directMarkerCandidate : void 0;
  const extensionCandidates = [];
  const searchStart = Math.max(0, leader.opIndex - 500);
  const searchEnd = Math.min(scene.ops.length - 1, leader.opIndex + 500);
  const extensionPool = /* @__PURE__ */ new Set();
  for (let opIndex = searchStart; opIndex <= searchEnd; opIndex += 1) extensionPool.add(opIndex);
  if (spatialIndex) {
    spatialIndex.queryPoint(target.x, target.y, spatialSearchRadius).forEach((opIndex) => extensionPool.add(opIndex));
  }
  for (const opIndex of [...extensionPool].sort((left, right) => left - right)) {
    if (outlineMembers.has(opIndex) || leader.pathOps.includes(opIndex)) continue;
    const op = scene.ops[opIndex];
    if (op.kind !== "path" || !op.stroke || op.fill || op.segments.length < 2 || op.segments.length > 4) continue;
    const opDiagonal = boundsDiagonal(op.bounds);
    if (opDiagonal < pageDiagonal * 1e-3 || opDiagonal > pageDiagonal * 0.32) continue;
    if (Math.abs(groupForOp(segmentation, opIndex) - leaderGroup) > 24) continue;
    const endpoints = pathEndpoints(op);
    if (!endpoints) continue;
    const firstGap = pointDistance(target, endpoints[0]);
    const secondGap = pointDistance(target, endpoints[1]);
    const connection = firstGap <= secondGap ? endpoints[0] : endpoints[1];
    const other = connection === endpoints[0] ? endpoints[1] : endpoints[0];
    const connectionGap = Math.min(firstGap, secondGap);
    const extensionDx = other.x - connection.x;
    const extensionDy = other.y - connection.y;
    const extensionLength = Math.max(1e-9, Math.hypot(extensionDx, extensionDy));
    const directionalContinuation = (currentDx * extensionDx + currentDy * extensionDy) / (currentLength * extensionLength);
    const parallel = Math.abs(directionalContinuation);
    const normalConnection = Math.max(cluster.scale * 1.8, pageDiagonal * 3e-3);
    const collinearConnection = Math.max(cluster.scale * 3, pageDiagonal * 45e-4);
    const connectionThreshold = allowCrossBatchTopology ? endpointContactThreshold(
      pageDiagonal,
      cluster.scale,
      options.leaderConnectionScale,
      Math.max(leader.op.lineWidth, op.lineWidth),
      Math.min(opDiagonal, cluster.scale * 4)
    ) : parallel >= 0.985 ? collinearConnection : normalConnection;
    if (connectionGap > connectionThreshold) continue;
    const marker = nearestMarker(other);
    if (!marker) continue;
    const groupGap = Math.abs(groupForOp(segmentation, opIndex) - leaderGroup);
    const markerGroupGap = Math.abs(groupForOp(segmentation, marker.opIndex) - leaderGroup);
    const paintGap = Math.abs(opIndex - leader.opIndex);
    const spatiallyRecovered = opIndex < searchStart || opIndex > searchEnd || !localMarkerOps.has(marker.opIndex);
    const markerTopologyMatches = !allowCrossBatchTopology || spatialTriangleMarkerMatches(
      scene,
      marker.opIndex,
      [...leader.pathOps, opIndex],
      root,
      other,
      currentLength + extensionLength,
      pageDiagonal
    );
    if (!markerTopologyMatches) continue;
    const lineWidthFloor = pageDiagonal * 1e-5;
    const lineWidthRatio = Math.max(leader.op.lineWidth, op.lineWidth, lineWidthFloor) / Math.max(Math.min(leader.op.lineWidth, op.lineWidth), lineWidthFloor);
    const compatibleStrokeStyle = leader.op.strokeColor === op.strokeColor && Math.abs(leader.op.strokeAlpha - op.strokeAlpha) <= 0.05 && lineWidthRatio <= 1.1 && leader.op.dash.length === op.dash.length && leader.op.dash.every((value, index) => Math.abs(value - op.dash[index]) <= pageDiagonal * 1e-5);
    const exactJoinThreshold = Math.max(
      pageDiagonal * 1e-5,
      Math.max(leader.op.lineWidth, op.lineWidth) * 0.25
    );
    const trustedLocalMarkedElbow = allowCrossBatchTopology && Boolean(segmentation) && !spatiallyRecovered && groupGap === 0 && markerGroupGap === 0 && compatibleStrokeStyle && connectionGap <= exactJoinThreshold;
    if (!trustedLocalMarkedElbow && allowCrossBatchTopology && directionalContinuation < 0.3) continue;
    const extensionEscapePoint = trustedLocalMarkedElbow ? triangleArrowApex(
      scene,
      [marker.opIndex],
      [...leader.pathOps, opIndex],
      root,
      other,
      pageDiagonal
    ) ?? other : other;
    if (pointToBounds(extensionEscapePoint, cluster.bounds) < cluster.scale * 2.5) continue;
    const score = allowCrossBatchTopology ? (connectionGap + marker.distance) / Math.max(1e-9, connectionThreshold) + groupGap * 0.02 + clamp01(paintGap / Math.max(1, scene.ops.length)) * 0.02 : connectionGap + marker.distance + paintGap * 0.1 + groupGap * 0.1;
    extensionCandidates.push({
      opIndex,
      other,
      markerOp: marker.opIndex,
      score,
      spatiallyRecovered
    });
  }
  extensionCandidates.sort((left, right) => left.score - right.score);
  const localExtension = extensionCandidates.find((candidate) => !candidate.spatiallyRecovered);
  const extension = allowCrossBatchTopology && !localExtension ? (() => {
    const routeMergeThreshold = Math.max(pageDiagonal * 2e-5, cluster.scale * 0.05);
    const routeGroups = [];
    for (const candidate of extensionCandidates) {
      const group = routeGroups.find((items) => items[0].markerOp === candidate.markerOp && pointDistance(items[0].other, candidate.other) <= routeMergeThreshold);
      if (group) group.push(candidate);
      else routeGroups.push([candidate]);
    }
    return routeGroups.length === 1 ? routeGroups[0][0] : void 0;
  })() : localExtension ?? extensionCandidates[0];
  const directIsAtEndpoint = Boolean(directMarker && (allowCrossBatchTopology || directMarker.distance <= Math.max(cluster.scale * 0.25, pageDiagonal * 5e-4)));
  if (directIsAtEndpoint && directMarker) {
    const directArrowOps = leader.arrowheadOps.includes(directMarker.opIndex) ? [...leader.arrowheadOps] : [directMarker.opIndex];
    return {
      ...leader,
      endpoints: [root, target],
      arrowheadOps: directArrowOps,
      packetKind: "arrow-leader",
      markerKind: "filled-arrow",
      markerOps: [...directArrowOps],
      spatialEndpointRecovered: leader.spatialEndpointRecovered || allowCrossBatchTopology
    };
  }
  if (extension) {
    return {
      ...leader,
      pathOps: [.../* @__PURE__ */ new Set([...leader.pathOps, extension.opIndex])],
      endpoints: [root, extension.other],
      arrowheadOps: [extension.markerOp],
      packetKind: "arrow-leader",
      markerKind: "filled-arrow",
      markerOps: [extension.markerOp],
      spatialEndpointRecovered: leader.spatialEndpointRecovered || allowCrossBatchTopology || extension.spatiallyRecovered
    };
  }
  if (directMarker) {
    const directArrowOps = leader.arrowheadOps.includes(directMarker.opIndex) ? [...leader.arrowheadOps] : [directMarker.opIndex];
    return {
      ...leader,
      endpoints: [root, target],
      arrowheadOps: directArrowOps,
      packetKind: "arrow-leader",
      markerKind: "filled-arrow",
      markerOps: [...directArrowOps],
      spatialEndpointRecovered: leader.spatialEndpointRecovered || allowCrossBatchTopology
    };
  }
  return leader.markerKind === "open-marker" ? { ...leader, endpoints: [root, target], arrowheadOps: [] } : { ...leader, endpoints: [root, target], arrowheadOps: [], markerKind: "none", markerOps: [] };
};
var arrowlessStructuralArtifacts = (scene, leaders, segmentation, pageDiagonal) => {
  const artifacts = /* @__PURE__ */ new Set();
  const connectionThreshold = pageDiagonal * 12e-4;
  const spatialIndex = getSceneSpatialIndex(scene);
  for (const leader of leaders.filter((item) => item.packetKind === "leader-only")) {
    const ownOps = new Set(leader.pathOps);
    const connectedEnds = leader.endpoints.map((endpoint) => spatialIndex.queryPoint(endpoint.x, endpoint.y, connectionThreshold).some((opIndex) => {
      const op = scene.ops[opIndex];
      return !ownOps.has(opIndex) && op.kind === "path" && op.stroke && !op.fill && boundsDiagonal(op.bounds) >= connectionThreshold && pointToBounds(endpoint, finiteBounds(op.bounds)) <= connectionThreshold && pointToPath(endpoint, op) <= connectionThreshold;
    }));
    if (connectedEnds[0] && connectedEnds[1]) {
      artifacts.add(leader.opIndex);
      continue;
    }
    const [start, end] = leader.endpoints;
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.max(1e-9, Math.hypot(dx, dy));
    const group = groupForOp(segmentation, leader.opIndex);
    let parallelPeers = 0;
    const from = Math.max(0, leader.opIndex - 20);
    const to = Math.min(scene.ops.length - 1, leader.opIndex + 20);
    for (let opIndex = from; opIndex <= to; opIndex += 1) {
      if (ownOps.has(opIndex) || groupForOp(segmentation, opIndex) !== group) continue;
      const op = scene.ops[opIndex];
      if (op.kind !== "path" || !op.stroke || op.fill || op.segments.length !== 2) continue;
      const endpoints = pathEndpoints(op);
      if (!endpoints) continue;
      const peerDx = endpoints[1].x - endpoints[0].x;
      const peerDy = endpoints[1].y - endpoints[0].y;
      const peerLength = Math.hypot(peerDx, peerDy);
      if (peerLength < length * 0.35 || peerLength > length * 2.85) continue;
      const parallel = Math.abs(dx * peerDx + dy * peerDy) / (length * Math.max(1e-9, peerLength));
      if (parallel >= 0.997) parallelPeers += 1;
    }
    if (parallelPeers >= 2) artifacts.add(leader.opIndex);
  }
  return artifacts;
};
var isInlineCarrierLabel = (scene, cluster, leader, segmentation, pageDiagonal) => {
  if (cluster.source !== "decoded" || leader.packetKind !== "leader-only") return false;
  const normalized = cluster.text.replace(/\s+/g, " ").trim();
  if (cluster.opIndices.length !== 1 || normalized.length > 12) return false;
  const [first, second] = leader.endpoints;
  const root = pointToBounds(first, cluster.bounds) <= pointToBounds(second, cluster.bounds) ? first : second;
  const target = root === first ? second : first;
  const outgoing = { x: target.x - root.x, y: target.y - root.y };
  const outgoingLength = Math.hypot(outgoing.x, outgoing.y);
  if (outgoingLength <= 1e-9) return false;
  const axis = { x: outgoing.x / outgoingLength, y: outgoing.y / outgoingLength };
  const center = {
    x: (cluster.bounds.minX + cluster.bounds.maxX) / 2,
    y: (cluster.bounds.minY + cluster.bounds.maxY) / 2
  };
  const rootSide = (root.x - center.x) * axis.x + (root.y - center.y) * axis.y;
  const nearText = Math.max(cluster.scale * 2.5, pageDiagonal * 25e-4);
  const ownOps = new Set(leader.pathOps);
  const group = groupForOp(segmentation, leader.opIndex);
  const from = Math.max(0, Math.min(cluster.opIndices[0], leader.opIndex) - 12);
  const to = Math.min(scene.ops.length - 1, Math.max(cluster.opIndices.at(-1), leader.opIndex) + 12);
  for (let opIndex = from; opIndex <= to; opIndex += 1) {
    if (ownOps.has(opIndex) || cluster.opIndices.includes(opIndex) || groupForOp(segmentation, opIndex) !== group) continue;
    const op = scene.ops[opIndex];
    if (op.kind !== "path" || !op.stroke || op.fill || op.segments.length < 2 || op.segments.length > 4) continue;
    const endpoints = pathEndpoints(op);
    if (!endpoints) continue;
    const peerRoot = pointToBounds(endpoints[0], cluster.bounds) <= pointToBounds(endpoints[1], cluster.bounds) ? endpoints[0] : endpoints[1];
    const peerTarget = peerRoot === endpoints[0] ? endpoints[1] : endpoints[0];
    if (pointToBounds(peerRoot, cluster.bounds) > nearText || pointToBounds(peerTarget, cluster.bounds) <= pointToBounds(peerRoot, cluster.bounds) + cluster.scale * 0.5) continue;
    const peer = { x: peerTarget.x - peerRoot.x, y: peerTarget.y - peerRoot.y };
    const peerLength = Math.hypot(peer.x, peer.y);
    if (peerLength < outgoingLength * 0.2 || peerLength > outgoingLength * 4) continue;
    const parallel = Math.abs(outgoing.x * peer.x + outgoing.y * peer.y) / (outgoingLength * peerLength);
    if (parallel < 0.995) continue;
    const peerSide = (peerRoot.x - center.x) * axis.x + (peerRoot.y - center.y) * axis.y;
    if (rootSide * peerSide <= 0) return true;
  }
  return false;
};
var hasTitleUnderline = (scene, cluster, pageDiagonal) => {
  const width = boundsWidth(cluster.bounds);
  const height = boundsHeight(cluster.bounds);
  if (cluster.source !== "vector-outline" || width < height * 3) return false;
  const proximity = Math.max(cluster.scale * 1.5, pageDiagonal * 12e-4);
  const firstText = cluster.opIndices[0];
  const lastText = cluster.opIndices.at(-1);
  const from = Math.max(0, firstText - 3);
  const to = Math.min(scene.ops.length - 1, lastText + 3);
  for (let opIndex = from; opIndex <= to; opIndex += 1) {
    if (cluster.opIndices.includes(opIndex)) continue;
    const op = scene.ops[opIndex];
    if (op.kind !== "path" || !op.stroke || op.fill || op.segments.length !== 2) continue;
    const endpoints = pathEndpoints(op);
    if (!endpoints) continue;
    const lineWidth = Math.abs(endpoints[1].x - endpoints[0].x);
    const lineHeight = Math.abs(endpoints[1].y - endpoints[0].y);
    if (lineHeight > Math.max(pageDiagonal * 3e-4, lineWidth * 0.03)) continue;
    const horizontalOverlap = Math.max(
      0,
      Math.min(cluster.bounds.maxX, op.bounds.maxX) - Math.max(cluster.bounds.minX, op.bounds.minX)
    );
    const verticalGap = Math.max(
      0,
      cluster.bounds.minY - op.bounds.maxY,
      op.bounds.minY - cluster.bounds.maxY
    );
    if (horizontalOverlap >= width * 0.75 && lineWidth >= width * 0.75 && lineWidth <= width * 1.35 && verticalGap <= proximity) {
      return true;
    }
  }
  return false;
};
var excludedTextClass = (text) => {
  const normalized = text.replace(/\s+/g, " ").trim().toUpperCase();
  if (!/[\p{L}\p{N}]/u.test(normalized)) return "non-text-symbol";
  if (normalized === "NORTH") return "north-arrow";
  return null;
};
var subtypeFor = (text) => {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (/\b\d{1,3}\s*\/\s*[A-Z]\d{2,4}\b/i.test(normalized)) return "detail-reference";
  if (/^\s*[A-Z]?\d{1,3}\s*$/.test(normalized)) return "symbol-callout";
  return "note";
};
var fnv1a = (value) => {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
};
var groupForOp = (segmentation, opIndex) => segmentation ? Math.max(1, segmentation.assignments[opIndex] ?? 1) : 1;
var hasStrictDetailReference = (text) => /\b\d{1,3}\s*\/\s*[A-Z]\d{2,4}\b/i.test(text.replace(/\s+/g, " "));
var normalizeDetailTagLeaderNetworks = (scene, segmentation, pageDiagonal, inputTextClusters, inputLeaders) => {
  let textClusters = [...inputTextClusters];
  const suppressed = /* @__PURE__ */ new Set();
  const replacements = /* @__PURE__ */ new Map();
  const claimedContinuations = /* @__PURE__ */ new Set();
  const exactContact = Math.max(1e-4, pageDiagonal * 2e-5);
  const mergeDecodedClusters = (left, right) => {
    const ordered = [left, right].sort((a, b) => a.opIndices[0] - b.opIndices[0]);
    const opIndices = [...new Set(ordered.flatMap((cluster) => cluster.opIndices))].sort((a, b) => a - b);
    const ops = opIndices.map((opIndex) => scene.ops[opIndex]).filter((op) => op?.kind === "text");
    return {
      opIndices,
      ops,
      bounds: unionBounds(left.bounds, right.bounds),
      text: ordered.map((cluster) => cluster.text).join("\n"),
      scale: Math.max(left.scale, right.scale),
      source: "decoded"
    };
  };
  const stubs = inputLeaders.filter(
    (leader) => leader.packetKind === "leader-only" && leader.markerOps.length === 0 && leader.pathOps.length === 1 && leader.pathOps[0] === leader.opIndex && scene.ops[leader.opIndex + 2]?.kind === "text"
  ).sort((a, b) => a.opIndex - b.opIndex);
  for (const stub of stubs) {
    const stubGroup = groupForOp(segmentation, stub.opIndex);
    const cap = scene.ops[stub.opIndex + 1];
    if (cap?.kind !== "path" || !cap.fill || !cap.stroke || cap.segments.length < 8 || cap.segments.length > 10 || cap.segments.filter((segment) => segment.kind === "move").length !== 2 || cap.segments.filter((segment) => segment.kind === "close").length < 2 || groupForOp(segmentation, stub.opIndex + 1) !== stubGroup) continue;
    const capBounds = finiteBounds(cap.bounds);
    const capWidth = boundsWidth(capBounds);
    const capHeight = boundsHeight(capBounds);
    const capMinor = Math.max(1e-9, Math.min(capWidth, capHeight));
    if (boundsDiagonal(capBounds) > pageDiagonal * 8e-3 || Math.max(capWidth, capHeight) / capMinor < 2) continue;
    const capDistances = stub.endpoints.map((point) => pointToBounds(point, capBounds));
    const capEndIndex = capDistances[0] <= capDistances[1] ? 0 : 1;
    const junctionIndex = capEndIndex === 0 ? 1 : 0;
    const capEnd = stub.endpoints[capEndIndex];
    const junction = stub.endpoints[junctionIndex];
    const capTouchThreshold = Math.max(pageDiagonal * 35e-5, capMinor * 0.75);
    if (capDistances[capEndIndex] > capTouchThreshold || capDistances[junctionIndex] <= capTouchThreshold * 2) continue;
    const description = textClusters.find((cluster) => cluster.source === "decoded" && cluster.opIndices.includes(stub.opIndex + 2));
    if (!description || groupForOp(segmentation, description.opIndices[0]) !== stubGroup || pointToBounds(capEnd, description.bounds) > Math.max(description.scale * 0.75, pageDiagonal * 15e-4)) continue;
    let carrier = description;
    if (!hasStrictDetailReference(description.text)) {
      const referenceCandidates = textClusters.filter((cluster) => {
        if (cluster === description || cluster.source !== "decoded" || !hasStrictDetailReference(cluster.text) || groupForOp(segmentation, cluster.opIndices[0]) !== stubGroup) return false;
        const descriptionStart = description.opIndices[0];
        const descriptionEnd = description.opIndices.at(-1);
        const referenceStart = cluster.opIndices[0];
        const referenceEnd = cluster.opIndices.at(-1);
        const paintGap = referenceStart > descriptionEnd ? referenceStart - descriptionEnd - 1 : descriptionStart > referenceEnd ? descriptionStart - referenceEnd - 1 : 0;
        const firstDescriptionOp = description.ops[0];
        const firstReferenceOp = cluster.ops[0];
        return paintGap <= 1 && Math.max(descriptionEnd, referenceEnd) <= stub.opIndex + 8 && Boolean(firstDescriptionOp && firstReferenceOp && orientationDelta(firstDescriptionOp, firstReferenceOp) <= 0.12) && boundsGap(description.bounds, cluster.bounds) <= Math.max(
          pageDiagonal * 0.02,
          Math.max(description.scale, cluster.scale) * 4
        );
      });
      if (referenceCandidates.length !== 1) continue;
      carrier = mergeDecodedClusters(description, referenceCandidates[0]);
    }
    const continuations = inputLeaders.flatMap((leader) => {
      if (leader === stub || suppressed.has(leader) || claimedContinuations.has(leader) || leader.markerOps.length === 0 || leader.packetKind === "leader-only") return [];
      const groupGap = Math.abs(groupForOp(segmentation, leader.opIndex) - stubGroup);
      const paintGap = Math.abs(leader.opIndex - stub.opIndex);
      if (groupGap > 1 || paintGap > 24) return [];
      const junctionDistance = Math.min(...leader.pathOps.map((opIndex) => {
        const op = scene.ops[opIndex];
        return op?.kind === "path" ? pointToPath(junction, op) : Number.POSITIVE_INFINITY;
      }));
      const capEndDistance = Math.min(...leader.pathOps.map((opIndex) => {
        const op = scene.ops[opIndex];
        return op?.kind === "path" ? pointToPath(capEnd, op) : Number.POSITIVE_INFINITY;
      }));
      if (junctionDistance > exactContact || capEndDistance <= exactContact * 4) return [];
      return [{ leader, junctionDistance, groupGap, paintGap }];
    }).sort((left, right) => left.junctionDistance - right.junctionDistance || left.groupGap - right.groupGap || left.paintGap - right.paintGap || left.leader.opIndex - right.leader.opIndex);
    if (!continuations.length) {
      suppressed.add(stub);
      continue;
    }
    const continuation = continuations[0].leader;
    const markerEndpoint = [...continuation.endpoints].sort((left, right) => {
      const markerDistance = (point) => Math.min(...continuation.markerOps.map((opIndex) => {
        const marker = scene.ops[opIndex];
        return marker?.kind === "path" ? pointToBounds(point, finiteBounds(marker.bounds)) : Number.POSITIVE_INFINITY;
      }));
      return markerDistance(left) - markerDistance(right);
    })[0];
    const composite = {
      ...continuation,
      // Anchor proposal sequencing/group ownership at the carrier-side packet.
      opIndex: stub.opIndex,
      op: stub.op,
      pathOps: [.../* @__PURE__ */ new Set([...stub.pathOps, ...continuation.pathOps])].sort((a, b) => a - b),
      endpoints: [{ ...capEnd }, { ...markerEndpoint }],
      authoredTextOp: stub.opIndex + 2
    };
    suppressed.add(stub);
    suppressed.add(continuation);
    claimedContinuations.add(continuation);
    replacements.set(continuation, composite);
    if (carrier !== description) {
      const reference = textClusters.find((cluster) => cluster !== description && cluster.source === "decoded" && cluster.opIndices.some((opIndex) => carrier.opIndices.includes(opIndex)));
      textClusters = textClusters.filter((cluster) => cluster !== description && cluster !== reference);
      textClusters.push(carrier);
      textClusters.sort((a, b) => a.opIndices[0] - b.opIndices[0]);
    }
  }
  const leaders = inputLeaders.flatMap((leader) => {
    const replacement = replacements.get(leader);
    if (replacement) return [replacement];
    return suppressed.has(leader) ? [] : [leader];
  });
  return { textClusters, leaders };
};
function detectCalloutsInternal(scene, segmentation, partialOptions = {}, forcedOutlineTextRois = []) {
  const options = { ...DEFAULT_CALLOUT_DETECTION_OPTIONS, ...partialOptions };
  const pageBounds = finiteBounds(scene.pageBounds ?? scene.bounds ?? { minX: 0, minY: 0, maxX: 1, maxY: 1 });
  const pageDiagonal = Math.max(1, boundsDiagonal(pageBounds));
  const spatialIndex = getSceneSpatialIndex(scene);
  const decodedTextClusters = clusterTextOps(scene, pageDiagonal, options);
  const allOutlineTextClusters = forcedOutlineTextRois.length ? vectorOutlineTextClusters(scene, segmentation, pageDiagonal) : [];
  const relaxedOutlineTextRuns = forcedOutlineTextRois.length ? relaxedVectorOutlineTextRuns(scene, segmentation, pageDiagonal) : [];
  const forcedOutlineTextClusters = forcedOutlineTextRois.flatMap((roi) => {
    const target = finiteBounds(roi);
    const byGroup = /* @__PURE__ */ new Map();
    const addCluster = (cluster) => {
      const group = groupForOp(segmentation, cluster.opIndices[0]);
      const owned = byGroup.get(group) ?? [];
      owned.push(cluster);
      byGroup.set(group, owned);
    };
    for (const cluster of allOutlineTextClusters) {
      if (boundsOverlapFraction(cluster.bounds, target) < 0.45) continue;
      addCluster(cluster);
    }
    vectorOutlineTextClustersInRoi(scene, pageDiagonal, target, relaxedOutlineTextRuns).forEach(addCluster);
    return [...byGroup].map(([group, clusters]) => {
      const opIndices = [...new Set(clusters.flatMap((cluster) => cluster.opIndices))].sort((left, right) => left - right);
      const bounds = clusters.slice(1).reduce(
        (combined, cluster) => unionBounds(combined, cluster.bounds),
        finiteBounds(clusters[0].bounds)
      );
      return {
        opIndices,
        ops: [],
        bounds,
        text: `\u77E2\u91CF\u8F6E\u5ED3\u6587\u5B57 #${group}:${opIndices[0]}`,
        scale: Math.max(...clusters.map((cluster) => cluster.scale)),
        source: "vector-outline",
        roiSeeded: clusters.some((cluster) => cluster.roiSeeded)
      };
    });
  });
  const rawOutlineTextClusters = forcedOutlineTextRois.length ? forcedOutlineTextClusters : [];
  const allSymbolClusters = vectorSymbolClusters(scene, segmentation, pageDiagonal);
  const scopedCanonicalSymbolClusters = forcedOutlineTextRois.length ? allSymbolClusters.filter((cluster) => forcedOutlineTextRois.some((roi) => intersects2(cluster.bounds, finiteBounds(roi)) || boundsGap(cluster.bounds, finiteBounds(roi)) <= pageDiagonal * 6e-3)).map((cluster) => ({ ...cluster, roiSeeded: true })) : allSymbolClusters;
  const roiSymbolClusters = vectorSymbolClustersInRois(
    scene,
    segmentation,
    pageDiagonal,
    forcedOutlineTextRois
  ).filter((cluster) => !scopedCanonicalSymbolClusters.some((canonical) => cluster.opIndices.some((opIndex) => canonical.opIndices.includes(opIndex))));
  const symbolClusters = [...scopedCanonicalSymbolClusters, ...roiSymbolClusters].sort((left, right) => left.opIndices[0] - right.opIndices[0]);
  const outlineTextClusters = mergeOutlineTextWithSymbols(
    rawOutlineTextClusters,
    symbolClusters,
    segmentation,
    pageDiagonal
  );
  const scopedDecodedTextClusters = forcedOutlineTextRois.length ? decodedTextClusters.filter((cluster) => forcedOutlineTextRois.some((roi) => intersects2(cluster.bounds, finiteBounds(roi)))) : decodedTextClusters;
  const symbolMembers = new Set(symbolClusters.flatMap((cluster) => cluster.opIndices));
  const authoredLeaders = leaderCandidates(scene, pageDiagonal, segmentation, symbolMembers);
  const outlineMembers = new Set(outlineTextClusters.flatMap((cluster) => cluster.opIndices));
  const rawLeaders = [
    ...authoredLeaders,
    ...outlineLeaderCandidates(scene, outlineTextClusters, authoredLeaders, pageDiagonal)
  ].filter((leader) => !leader.pathOps.some((opIndex) => outlineMembers.has(opIndex)));
  const rawTextClusters = [
    ...scopedDecodedTextClusters,
    ...outlineTextClusters
  ].sort((left, right) => left.opIndices[0] - right.opIndices[0]);
  const normalizedDetailNetworks = normalizeDetailTagLeaderNetworks(
    scene,
    segmentation,
    pageDiagonal,
    rawTextClusters,
    rawLeaders
  );
  const leaders = normalizedDetailNetworks.leaders;
  const textClusters = normalizedDetailNetworks.textClusters;
  const arrowheads = arrowheadCandidates(scene, pageDiagonal);
  const authoredMarkedPackets = authoredLeaders.flatMap((leader) => leader.packetKind !== "leader-only" && leader.markerOps.length ? [[.../* @__PURE__ */ new Set([...leader.pathOps, ...leader.markerOps])]] : []);
  const conflictsWithAuthoredPacket = (candidateOps) => {
    const candidate = new Set(candidateOps);
    return authoredMarkedPackets.some((packet) => packet.some((opIndex) => candidate.has(opIndex)) && !packet.every((opIndex) => candidate.has(opIndex)));
  };
  const structuralArtifacts = arrowlessStructuralArtifacts(scene, leaders, segmentation, pageDiagonal);
  const shortTextFrequency = /* @__PURE__ */ new Map();
  for (const cluster of textClusters) {
    const normalized = cluster.text.replace(/\s+/g, " ").trim().toUpperCase();
    if (normalized.length > 3) continue;
    const key = `${groupForOp(segmentation, cluster.opIndices[0])}|${normalized}`;
    shortTextFrequency.set(key, (shortTextFrequency.get(key) ?? 0) + 1);
  }
  const claimedLeaders = /* @__PURE__ */ new Set();
  const callouts = [];
  const rawProposals = leaders.flatMap((leader) => textClusters.flatMap((cluster) => {
    const symbolCarrier = cluster.carrierKind === "symbol" || cluster.carrierKind === "text-symbol";
    const rootThreshold = cluster.source === "vector-outline" ? Math.min(
      pageDiagonal * 0.02,
      Math.max(cluster.scale * 3.8, pageDiagonal * 25e-4)
    ) : Math.min(
      pageDiagonal * 0.018,
      Math.max(cluster.scale * options.leaderRootDistanceScale, pageDiagonal * 25e-4)
    );
    const clusterGroup = groupForOp(segmentation, cluster.symbolOpIndex ?? cluster.opIndices[0]);
    const initialLeaderGroup = groupForOp(segmentation, leader.opIndex);
    const initialGroupGap = Math.abs(clusterGroup - initialLeaderGroup);
    const carrierBounds = carrierBoundsForCluster(scene, cluster);
    const initialRootDistance = Math.min(
      pointToBounds(leader.endpoints[0], carrierBounds),
      pointToBounds(leader.endpoints[1], carrierBounds)
    );
    if (initialGroupGap > (symbolCarrier ? 1 : cluster.source === "vector-outline" ? 24 : 1) || initialRootDistance > rootThreshold) return [];
    const expandedLeader = expandOutlineLeader(
      scene,
      leader,
      cluster,
      arrowheads,
      outlineMembers,
      segmentation,
      pageDiagonal,
      options
    );
    const firstDistance = pointToBounds(expandedLeader.endpoints[0], carrierBounds);
    const secondDistance = pointToBounds(expandedLeader.endpoints[1], carrierBounds);
    const rootDistance = Math.min(firstDistance, secondDistance);
    const rootPoint = firstDistance <= secondDistance ? expandedLeader.endpoints[0] : expandedLeader.endpoints[1];
    const connectionPoint = firstDistance <= secondDistance ? expandedLeader.endpoints[1] : expandedLeader.endpoints[0];
    const semanticTarget = expandedLeader.spatialEndpointRecovered && expandedLeader.markerOps.length ? resolveLeaderTarget(scene, expandedLeader, rootPoint, connectionPoint, pageDiagonal) : void 0;
    const targetDistance = Math.max(firstDistance, secondDistance);
    const semanticEscapeDistance = semanticTarget?.terminalKind === "arrow-apex" ? Math.max(targetDistance, pointToBounds(semanticTarget, carrierBounds)) : targetDistance;
    const carrierWidth = boundsWidth(carrierBounds);
    const carrierHeight = boundsHeight(carrierBounds);
    const rootMinorAxisDistance = carrierHeight >= carrierWidth * 1.4 ? Math.max(0, carrierBounds.minX - rootPoint.x, rootPoint.x - carrierBounds.maxX) : carrierWidth >= carrierHeight * 1.4 ? Math.max(0, carrierBounds.minY - rootPoint.y, rootPoint.y - carrierBounds.maxY) : 0;
    const spatialRootAligned = !expandedLeader.spatialEndpointRecovered || rootMinorAxisDistance <= Math.max(cluster.scale * 0.75, pageDiagonal * 4e-4);
    const spatialAuthoredConflict = Boolean(expandedLeader.spatialEndpointRecovered && conflictsWithAuthoredPacket([...expandedLeader.pathOps, ...expandedLeader.markerOps]));
    const firstText = cluster.opIndices[0];
    const lastText = cluster.opIndices.at(-1);
    const paintGap = expandedLeader.opIndex < firstText ? firstText - expandedLeader.opIndex : expandedLeader.opIndex > lastText ? expandedLeader.opIndex - lastText : 0;
    const geometryScore = clamp01(1 - rootDistance / Math.max(1e-9, rootThreshold));
    const sequenceScore = clamp01(1 - paintGap / Math.max(1, options.maxLocalPaintGap));
    const leaderGroup = groupForOp(segmentation, expandedLeader.opIndex);
    const groupGap = Math.abs(clusterGroup - leaderGroup);
    const localGroupContext = cluster.source === "vector-outline" ? groupGap <= (symbolCarrier ? 1 : 24) : groupGap <= 1;
    const leaderAfterText = expandedLeader.opIndex > lastText;
    const packetGap = leaderAfterText ? expandedLeader.opIndex - lastText : firstText - expandedLeader.opIndex;
    const packetOrder = cluster.source === "vector-outline" ? !cluster.roiSeeded || expandedLeader.packetKind !== "leader-only" || groupGap === 0 : expandedLeader.packetKind === "arrow-leader" ? packetGap >= 1 && packetGap <= 10 : expandedLeader.packetKind === "open-marker-leader" ? packetGap === expandedLeader.pathOps.length : packetGap >= 1 && packetGap <= 2;
    const betweenStart = leaderAfterText ? lastText + 1 : expandedLeader.opIndex + 1;
    const betweenEnd = leaderAfterText ? expandedLeader.opIndex : firstText;
    const interveningText = cluster.source === "decoded" && scene.ops.slice(betweenStart, betweenEnd).some((op) => op.kind === "text" && !cluster.ops.includes(op));
    const normalizedText = cluster.text.replace(/\s+/g, " ").trim().toUpperCase();
    const arrowlessNonCallout = expandedLeader.packetKind !== "arrow-leader" && (/^\s*(?:R\s*)?[\d.,+\-]+(?:\s*[- ]\s*\d+)?\s*['"]?\s*$/i.test(cluster.text) || /^[A-Z]{1,3}\d{1,4}$/.test(normalizedText));
    const repeatedSymbol = normalizedText.length <= 3 && (shortTextFrequency.get(`${clusterGroup}|${normalizedText}`) ?? 0) >= 4;
    const dimensionCore = normalizedText.replace(
      /(?:\(\s*TYP(?:ICAL)?\.?\s*\)|\bTYP(?:ICAL)?\.?)\s*$/i,
      ""
    ).trim();
    const dimensionMeasurement = cluster.source === "decoded" && /['"′″°]/.test(dimensionCore) && /^[\d\s.,+\-/'"′″°±]+$/.test(dimensionCore);
    const symbolRootThreshold = Math.max(cluster.scale, pageDiagonal * 275e-5);
    const closestOtherSymbol = symbolCarrier ? Math.min(...textClusters.flatMap((otherCluster) => {
      if (otherCluster === cluster || otherCluster.carrierKind !== "symbol" && otherCluster.carrierKind !== "text-symbol" || otherCluster.symbolOpIndex === void 0 || Math.abs(groupForOp(segmentation, otherCluster.symbolOpIndex) - leaderGroup) > 1) return [];
      const otherBounds = carrierBoundsForCluster(scene, otherCluster);
      return [Math.min(
        pointToBounds(expandedLeader.endpoints[0], otherBounds),
        pointToBounds(expandedLeader.endpoints[1], otherBounds)
      )];
    })) : Number.POSITIVE_INFINITY;
    const ambiguousSymbolOwner = symbolCarrier && closestOtherSymbol <= symbolRootThreshold && closestOtherSymbol < rootDistance + pageDiagonal * 7e-4;
    const markedSymbolPacket = expandedLeader.arrowheadOps.length > 0 || expandedLeader.packetKind === "open-marker-leader" && expandedLeader.pathOps.length >= 3;
    const symbolPacket = !symbolCarrier || markedSymbolPacket || cluster.roiSeeded;
    const geometricLeaderEscape = symbolCarrier ? semanticEscapeDistance >= Math.max(cluster.scale * 0.45, pageDiagonal * 12e-4) : semanticEscapeDistance >= rootThreshold * 0.65;
    const detailReferenceEscape = !symbolCarrier && subtypeFor(cluster.text) === "detail-reference";
    const leaderEscapesText = geometricLeaderEscape || detailReferenceEscape;
    const explicitRoiLeaderPacket = cluster.roiSeeded && cluster.source === "vector-outline" && expandedLeader.packetKind === "leader-only" && groupGap === 0 && paintGap <= 2 && rootDistance <= Math.max(cluster.scale * 0.75, pageDiagonal * 15e-4);
    const structuralArtifact = structuralArtifacts.has(leader.opIndex) && !explicitRoiLeaderPacket && (!(symbolCarrier && rootDistance <= symbolRootThreshold) && (cluster.source === "decoded" || expandedLeader.packetKind === "leader-only" && groupGap <= 2));
    const openMarkerTextMismatch = cluster.source === "decoded" && expandedLeader.packetKind === "open-marker-leader" && expandedLeader.authoredTextOp !== void 0 && !cluster.opIndices.includes(expandedLeader.authoredTextOp);
    const localPlausiblePair = packetOrder && !interveningText && rootDistance <= rootThreshold && localGroupContext;
    const inlineCarrier = localPlausiblePair && isInlineCarrierLabel(scene, cluster, expandedLeader, segmentation, pageDiagonal);
    const [leaderStart, leaderEnd] = expandedLeader.endpoints;
    const leaderDx = Math.abs(leaderEnd.x - leaderStart.x);
    const leaderDy = Math.abs(leaderEnd.y - leaderStart.y);
    const titleUnderline = !symbolCarrier && hasTitleUnderline(scene, cluster, pageDiagonal) && leaderDy <= leaderDx * 0.25;
    const passesExceptEscape = !excludedTextClass(cluster.text) && !arrowlessNonCallout && !repeatedSymbol && !dimensionMeasurement && !structuralArtifact && !openMarkerTextMismatch && !inlineCarrier && !titleUnderline && packetOrder && !interveningText && rootDistance <= rootThreshold && (!symbolCarrier || rootDistance <= symbolRootThreshold) && spatialRootAligned && !spatialAuthoredConflict && symbolPacket && !ambiguousSymbolOwner && localGroupContext;
    const qualifies = passesExceptEscape && leaderEscapesText;
    return [{
      leader: expandedLeader,
      cluster,
      rootDistance,
      targetDistance,
      rootThreshold,
      paintGap,
      geometryScore,
      sequenceScore,
      groupGap,
      passesExceptEscape,
      geometricLeaderEscape,
      detailReferenceEscape,
      qualifies
    }];
  }));
  const qualifyingAuthoredOwners = /* @__PURE__ */ new Map();
  for (const proposal of rawProposals) {
    if (!proposal.qualifies || proposal.leader.spatialEndpointRecovered || !proposal.leader.markerOps.length) continue;
    for (const opIndex of [...proposal.leader.pathOps, ...proposal.leader.markerOps]) {
      const owners = qualifyingAuthoredOwners.get(opIndex) ?? [];
      if (!owners.includes(proposal.cluster)) owners.push(proposal.cluster);
      qualifyingAuthoredOwners.set(opIndex, owners);
    }
  }
  const conflictsWithQualifyingAuthoredOwner = (candidateOps, candidateCluster) => candidateOps.some((opIndex) => (qualifyingAuthoredOwners.get(opIndex) ?? []).some((owner) => !owner.opIndices.some((carrierOp) => candidateCluster.opIndices.includes(carrierOp)) && boundsOverlapFraction(owner.bounds, candidateCluster.bounds) < 0.25 && boundsOverlapFraction(candidateCluster.bounds, owner.bounds) < 0.25 && boundsGap(owner.bounds, candidateCluster.bounds) > Math.max(
    pageDiagonal * 2e-4,
    Math.min(owner.scale, candidateCluster.scale) * 0.5
  )));
  const spatialMarkerOwners = /* @__PURE__ */ new Map();
  for (const proposal of rawProposals) {
    if (!proposal.qualifies || !proposal.leader.spatialEndpointRecovered) continue;
    const owner = {
      clusterOps: new Set(proposal.cluster.opIndices),
      pathOps: new Set(proposal.leader.pathOps)
    };
    for (const markerOp of proposal.leader.arrowheadOps) {
      const owners = spatialMarkerOwners.get(markerOp) ?? [];
      owners.push(owner);
      spatialMarkerOwners.set(markerOp, owners);
    }
  }
  const ambiguousSpatialMarkers = new Set(
    [...spatialMarkerOwners].flatMap(([markerOp, owners]) => {
      const carrierOverlaps = (left, right) => [...left.clusterOps].some((opIndex) => right.clusterOps.has(opIndex));
      const isSubset = (left, right) => [...left].every((opIndex) => right.has(opIndex));
      const maximal = owners.filter((owner, ownerIndex) => !owners.some((other, otherIndex) => ownerIndex !== otherIndex && carrierOverlaps(owner, other) && owner.pathOps.size < other.pathOps.size && isSubset(owner.pathOps, other.pathOps)));
      const canonical = [];
      for (const owner of maximal) {
        const duplicate = canonical.some((other) => carrierOverlaps(owner, other) && owner.pathOps.size === other.pathOps.size && isSubset(owner.pathOps, other.pathOps));
        if (!duplicate) canonical.push(owner);
      }
      return canonical.length > 1 ? [markerOp] : [];
    })
  );
  const proposalScore = (proposal) => proposal.cluster.source === "vector-outline" ? proposal.geometryScore * 0.5 + (proposal.leader.packetKind === "arrow-leader" ? 0.2 : 0) + (proposal.leader.packetKind === "arrow-leader" && proposal.paintGap <= 10 ? 0.03 : 0) + clamp01(1 - proposal.groupGap / 24) * 0.05 : proposal.geometryScore * 0.62 + proposal.sequenceScore * 0.38;
  const authoredMarkedPriority = (proposal) => !proposal.leader.spatialEndpointRecovered && proposal.leader.markerOps.length > 0 ? 1 : 0;
  const proposals = rawProposals.filter((proposal) => proposal.qualifies && (!proposal.leader.spatialEndpointRecovered || proposal.leader.arrowheadOps.every((markerOp) => !ambiguousSpatialMarkers.has(markerOp)) && !conflictsWithQualifyingAuthoredOwner(
    [...proposal.leader.pathOps, ...proposal.leader.markerOps],
    proposal.cluster
  ))).sort((left, right) => authoredMarkedPriority(right) - authoredMarkedPriority(left) || proposalScore(right) - proposalScore(left));
  const escapeBlockedByCluster = /* @__PURE__ */ new Map();
  for (const proposal of rawProposals) {
    const symbolCarrier = proposal.cluster.carrierKind === "symbol" || proposal.cluster.carrierKind === "text-symbol";
    if (proposal.cluster.source !== "decoded" || symbolCarrier || !proposal.passesExceptEscape || proposal.geometricLeaderEscape || proposal.detailReferenceEscape) continue;
    const blocked = escapeBlockedByCluster.get(proposal.cluster) ?? [];
    blocked.push(proposal);
    escapeBlockedByCluster.set(proposal.cluster, blocked);
  }
  const leadersByCluster = /* @__PURE__ */ new Map();
  for (const proposal of proposals) {
    const proposedMembers = [...proposal.leader.pathOps, ...proposal.leader.arrowheadOps];
    if (proposedMembers.some((opIndex) => claimedLeaders.has(opIndex))) continue;
    const owned = leadersByCluster.get(proposal.cluster) ?? [];
    if (proposal.cluster.source === "vector-outline" && owned.length > 0) {
      const symbolCarrier = proposal.cluster.carrierKind === "symbol" || proposal.cluster.carrierKind === "text-symbol";
      if (symbolCarrier) {
        const rootThreshold = Math.max(proposal.cluster.scale, pageDiagonal * 275e-5);
        const [first, second] = proposal.leader.endpoints;
        const proposedTarget = pointToBounds(first, proposal.cluster.bounds) >= pointToBounds(second, proposal.cluster.bounds) ? first : second;
        const existingTargets = owned.map((item) => {
          const [ownedFirst, ownedSecond] = item.leader.endpoints;
          return pointToBounds(ownedFirst, proposal.cluster.bounds) >= pointToBounds(ownedSecond, proposal.cluster.bounds) ? ownedFirst : ownedSecond;
        });
        const distinctTarget = existingTargets.every((target) => pointDistance(target, proposedTarget) > Math.max(
          proposal.cluster.scale * 0.8,
          pageDiagonal * 8e-4
        ));
        if (owned.length >= 6 || proposal.rootDistance > rootThreshold || !distinctTarget) continue;
      } else {
        const packetOps = (item) => [
          ...item.leader.arrowheadOps,
          ...item.leader.pathOps
        ];
        const existingPacketOps = new Set(owned.flatMap(packetOps));
        const proposedPacketOps = packetOps(proposal);
        const combinedPacketOps = [.../* @__PURE__ */ new Set([...existingPacketOps, ...proposedPacketOps])].sort((left, right) => left - right);
        const contiguousPacket = combinedPacketOps.every((opIndex, index) => index === 0 || opIndex === combinedPacketOps[index - 1] + 1);
        const firstText = proposal.cluster.opIndices[0];
        const lastText = proposal.cluster.opIndices.at(-1);
        const adjacentToText = combinedPacketOps[0] === lastText + 1 || combinedPacketOps.at(-1) === firstText - 1;
        const connectionThreshold = Math.max(
          proposal.cluster.scale * 0.75,
          pageDiagonal * 75e-5
        );
        const rootVicinity = Math.max(proposal.rootThreshold, connectionThreshold * 2);
        const proposedVertices = proposal.leader.pathOps.flatMap((opIndex) => {
          const op = scene.ops[opIndex];
          return op?.kind === "path" ? pathVertices(op) : [];
        }).filter((point) => pointToBounds(point, proposal.cluster.bounds) <= rootVicinity);
        const existingVertices = owned.flatMap((item) => item.leader.pathOps.flatMap((opIndex) => {
          const op = scene.ops[opIndex];
          return op?.kind === "path" ? pathVertices(op) : [];
        })).filter((point) => pointToBounds(point, proposal.cluster.bounds) <= rootVicinity);
        const connectedBranches = proposedVertices.some((point) => existingVertices.some((other) => pointDistance(point, other) <= connectionThreshold));
        const authoredMultiBranch = owned.length < 4 && owned.every((item) => item.leader.packetKind === "arrow-leader" && item.groupGap === 0 && item.paintGap <= 10) && proposal.leader.packetKind === "arrow-leader" && proposal.groupGap === 0 && proposal.paintGap <= 10 && proposedPacketOps.every((opIndex) => !existingPacketOps.has(opIndex)) && contiguousPacket && adjacentToText && connectedBranches;
        if (!authoredMultiBranch) continue;
      }
    }
    proposedMembers.forEach((opIndex) => claimedLeaders.add(opIndex));
    owned.push(proposal);
    leadersByCluster.set(proposal.cluster, owned);
  }
  for (const [cluster, blocked] of escapeBlockedByCluster) {
    const owned = leadersByCluster.get(cluster);
    if (!owned?.length || owned.length >= 4) continue;
    const normalSiblings = owned.filter((proposal) => proposal.qualifies && proposal.geometricLeaderEscape && proposal.leader.packetKind === "arrow-leader" && proposal.leader.arrowheadOps.length > 0 && proposal.groupGap === 0 && proposal.paintGap <= 10);
    if (!normalSiblings.length) continue;
    const carrierBounds = carrierBoundsForCluster(scene, cluster);
    const firstText = cluster.opIndices[0];
    const lastText = cluster.opIndices.at(-1);
    const connectionThreshold = Math.max(cluster.scale * 0.75, pageDiagonal * 75e-5);
    const exactContactThreshold = Math.max(cluster.scale * 0.08, pageDiagonal * 5e-5);
    const markerThreshold = Math.max(pageDiagonal * 35e-5, cluster.scale * 0.25);
    const targetSeparation = Math.max(cluster.scale * 0.8, pageDiagonal * 8e-4);
    const escapeIncrement = Math.max(cluster.scale * 0.25, pageDiagonal * 25e-5);
    const packetOps = (proposal) => [.../* @__PURE__ */ new Set([
      ...proposal.leader.arrowheadOps,
      ...proposal.leader.pathOps
    ])].sort((left, right) => left - right);
    const rootAndTarget = (proposal) => {
      const [first, second] = proposal.leader.endpoints;
      const firstDistance = pointToBounds(first, carrierBounds);
      const root = firstDistance <= pointToBounds(second, carrierBounds) ? first : second;
      const connection = root === first ? second : first;
      return {
        root,
        connection,
        target: resolveLeaderTarget(scene, proposal.leader, root, connection, pageDiagonal)
      };
    };
    for (const candidate of blocked.sort((left, right) => proposalScore(right) - proposalScore(left))) {
      if (owned.length >= 4 || candidate.leader.packetKind !== "arrow-leader" || candidate.leader.arrowheadOps.length === 0 || candidate.groupGap !== 0 || candidate.paintGap > 10) continue;
      const candidateOps = packetOps(candidate);
      if (candidateOps.some((opIndex) => claimedLeaders.has(opIndex))) continue;
      const candidateEnds = rootAndTarget(candidate);
      if (candidate.targetDistance - candidate.rootDistance < escapeIncrement) continue;
      const candidateMarkerBounds = candidate.leader.arrowheadOps.map((opIndex) => finiteBounds(scene.ops[opIndex].bounds)).reduce((combined, bounds) => unionBounds(combined, bounds));
      if (pointToBounds(candidateEnds.connection, candidateMarkerBounds) > markerThreshold) continue;
      const sibling = normalSiblings.find((normal) => {
        const siblingOps = packetOps(normal);
        if (candidateOps.some((opIndex) => siblingOps.includes(opIndex))) return false;
        const combinedOps = [...candidateOps, ...siblingOps].sort((left, right) => left - right);
        const contiguousPacket = combinedOps.every((opIndex, index) => index === 0 || opIndex === combinedOps[index - 1] + 1);
        const adjacentToText = combinedOps[0] === lastText + 1 || combinedOps.at(-1) === firstText - 1;
        if (!contiguousPacket || !adjacentToText) return false;
        const siblingEnds = rootAndTarget(normal);
        if (pointDistance(candidateEnds.root, siblingEnds.root) > connectionThreshold || pointDistance(candidateEnds.connection, siblingEnds.connection) <= connectionThreshold * 2) return false;
        const candidateJoinsSiblingPath = normal.leader.pathOps.some((opIndex) => {
          const op = scene.ops[opIndex];
          return op?.kind === "path" && pointToPath(candidateEnds.root, op) <= exactContactThreshold;
        });
        if (!candidateJoinsSiblingPath) return false;
        const siblingMarkerBounds = normal.leader.arrowheadOps.map((opIndex) => finiteBounds(scene.ops[opIndex].bounds)).reduce((combined, bounds) => unionBounds(combined, bounds));
        if (pointToBounds(siblingEnds.connection, siblingMarkerBounds) > markerThreshold) return false;
        return pointDistance(candidateEnds.target, siblingEnds.target) > targetSeparation && pointDistance(candidateEnds.connection, siblingEnds.connection) > targetSeparation;
      });
      if (!sibling) continue;
      candidateOps.forEach((opIndex) => claimedLeaders.add(opIndex));
      owned.push(candidate);
    }
  }
  for (const [cluster, owned] of leadersByCluster) {
    if (cluster.source !== "vector-outline" || cluster.carrierKind === "symbol" || cluster.carrierKind === "text-symbol" || owned.length >= 4) continue;
    let addedBranch = true;
    while (addedBranch && owned.length < 4) {
      addedBranch = false;
      const connectionThreshold = Math.max(
        cluster.scale * 0.35,
        pageDiagonal * 25e-5
      );
      const exactConnectionThreshold = Math.max(
        cluster.scale * 0.08,
        pageDiagonal * 5e-5
      );
      const distinctTargetThreshold = Math.max(
        cluster.scale * 2,
        pageDiagonal * 2e-3
      );
      const candidates = leaders.flatMap((leader) => {
        if (structuralArtifacts.has(leader.opIndex)) return [];
        const expandedLeader = expandOutlineLeader(
          scene,
          leader,
          cluster,
          arrowheads,
          outlineMembers,
          segmentation,
          pageDiagonal,
          options
        );
        if (expandedLeader.packetKind !== "arrow-leader") return [];
        if (expandedLeader.spatialEndpointRecovered && conflictsWithAuthoredPacket([...expandedLeader.pathOps, ...expandedLeader.markerOps])) return [];
        if (expandedLeader.spatialEndpointRecovered && conflictsWithQualifyingAuthoredOwner(
          [...expandedLeader.pathOps, ...expandedLeader.markerOps],
          cluster
        )) return [];
        if (expandedLeader.spatialEndpointRecovered && expandedLeader.arrowheadOps.some((markerOp) => ambiguousSpatialMarkers.has(markerOp))) return [];
        const members = [...expandedLeader.pathOps, ...expandedLeader.arrowheadOps];
        if (members.some((opIndex) => claimedLeaders.has(opIndex))) return [];
        const leaderGroup = groupForOp(segmentation, expandedLeader.opIndex);
        const endpointConnections = expandedLeader.endpoints.map((point) => owned.flatMap(
          (owner) => owner.leader.pathOps.map((opIndex) => {
            const op = scene.ops[opIndex];
            return {
              point,
              owner,
              distance: op?.kind === "path" ? pointToPath(point, op) : Number.POSITIVE_INFINITY
            };
          })
        ).sort((left, right) => left.distance - right.distance)[0]).sort((left, right) => left.distance - right.distance);
        const [connection, freeEnd] = endpointConnections;
        if (!connection || !freeEnd || connection.distance > connectionThreshold || freeEnd.distance <= connectionThreshold * 2) return [];
        const ownerGroup = groupForOp(segmentation, connection.owner.leader.opIndex);
        const leaderGroupGap = Math.abs(ownerGroup - leaderGroup);
        if (leaderGroupGap > 12) return [];
        const packetOps = (item) => [...item.pathOps, ...item.arrowheadOps].sort((left, right) => left - right);
        const candidatePacket = packetOps(expandedLeader);
        const ownerPacket = packetOps(connection.owner.leader);
        const packetGap = Math.min(...candidatePacket.flatMap((candidateOp) => ownerPacket.map((ownerOp) => Math.max(0, Math.abs(candidateOp - ownerOp) - 1))));
        if (leaderGroup === ownerGroup ? packetGap > 1 && connection.distance > exactConnectionThreshold : connection.distance > exactConnectionThreshold) return [];
        const markerLineWidth = Math.max(expandedLeader.op.lineWidth, ...expandedLeader.pathOps.map((opIndex) => {
          const op = scene.ops[opIndex];
          return op?.kind === "path" ? op.lineWidth : 0;
        }));
        const arrowAtFreeEnd = markerBundleTouchesEndpoint(
          scene,
          expandedLeader.arrowheadOps,
          [freeEnd.point, connection.point],
          pageDiagonal,
          markerLineWidth
        ) && expandedLeader.arrowheadOps.some((opIndex) => {
          const op = scene.ops[opIndex];
          if (op?.kind !== "path") return false;
          const threshold = endpointContactThreshold(
            pageDiagonal,
            cluster.scale,
            options.leaderConnectionScale,
            Math.max(markerLineWidth, op.lineWidth),
            boundsDiagonal(finiteBounds(op.bounds))
          );
          return pointToPath(freeEnd.point, op) <= threshold;
        });
        if (!arrowAtFreeEnd) return [];
        const ownedTargets = owned.map((item) => {
          const [first, second] = item.leader.endpoints;
          return pointToBounds(first, cluster.bounds) >= pointToBounds(second, cluster.bounds) ? first : second;
        });
        if (ownedTargets.some((target) => pointDistance(target, freeEnd.point) <= distinctTargetThreshold)) return [];
        const competingNetwork = [...leadersByCluster].some(([otherCluster, otherOwned]) => {
          if (otherCluster === cluster || otherCluster.source !== "vector-outline") return false;
          return otherOwned.some((otherOwner) => {
            const otherGroup = groupForOp(segmentation, otherOwner.leader.opIndex);
            if (Math.abs(otherGroup - leaderGroup) > 12) return false;
            return expandedLeader.endpoints.some((endpoint) => otherOwner.leader.pathOps.some((opIndex) => {
              const op = scene.ops[opIndex];
              return op?.kind === "path" && pointToPath(endpoint, op) <= exactConnectionThreshold;
            }));
          });
        });
        if (competingNetwork) return [];
        const rootDistance = pointToBounds(connection.point, cluster.bounds);
        const targetDistance = pointToBounds(freeEnd.point, cluster.bounds);
        if (targetDistance < owned[0].rootThreshold * 0.65) return [];
        return [{
          leader: expandedLeader,
          cluster,
          rootDistance,
          targetDistance,
          rootThreshold: owned[0].rootThreshold,
          paintGap: Math.min(
            Math.abs(leader.opIndex - cluster.opIndices[0]),
            Math.abs(leader.opIndex - cluster.opIndices.at(-1))
          ),
          geometryScore: clamp01(1 - connection.distance / connectionThreshold),
          sequenceScore: 0,
          groupGap: Math.abs(groupForOp(segmentation, cluster.symbolOpIndex ?? cluster.opIndices[0]) - leaderGroup),
          passesExceptEscape: true,
          geometricLeaderEscape: true,
          detailReferenceEscape: false,
          qualifies: true,
          connectionDistance: connection.distance
        }];
      }).sort((left, right) => left.connectionDistance - right.connectionDistance || left.groupGap - right.groupGap || left.leader.opIndex - right.leader.opIndex);
      const branchOwnersByMarker = /* @__PURE__ */ new Map();
      for (const candidate of candidates) {
        if (!candidate.leader.spatialEndpointRecovered) continue;
        for (const markerOp of candidate.leader.arrowheadOps) {
          const routes = branchOwnersByMarker.get(markerOp) ?? [];
          routes.push(new Set(candidate.leader.pathOps));
          branchOwnersByMarker.set(markerOp, routes);
        }
      }
      const ambiguousBranchMarkers = new Set([...branchOwnersByMarker].flatMap(([markerOp, routes]) => {
        const maximal = routes.filter((route, routeIndex) => !routes.some((other, otherIndex) => routeIndex !== otherIndex && route.size < other.size && [...route].every((opIndex) => other.has(opIndex))));
        const canonical = [];
        for (const route of maximal) {
          if (!canonical.some((other) => route.size === other.size && [...route].every((opIndex) => other.has(opIndex)))) canonical.push(route);
        }
        return canonical.length > 1 ? [markerOp] : [];
      }));
      const branch = candidates.find((candidate) => !candidate.leader.spatialEndpointRecovered || candidate.leader.arrowheadOps.every((markerOp) => !ambiguousBranchMarkers.has(markerOp)));
      if (!branch) continue;
      [...branch.leader.pathOps, ...branch.leader.arrowheadOps].forEach((opIndex) => claimedLeaders.add(opIndex));
      owned.push(branch);
      addedBranch = true;
    }
  }
  for (const [cluster, owned] of leadersByCluster) {
    const symbolCarrier = cluster.carrierKind === "symbol" || cluster.carrierKind === "text-symbol";
    const arrowheadOps = /* @__PURE__ */ new Set();
    const detectedLeaders = [];
    for (const proposal of owned) {
      const [first, second] = proposal.leader.endpoints;
      const rootBounds = carrierBoundsForCluster(scene, cluster);
      const firstDistance = pointToBounds(first, rootBounds);
      const root = firstDistance <= pointToBounds(second, rootBounds) ? first : second;
      const leaderEnd = root === first ? second : first;
      const attachedArrowOps = [...proposal.leader.arrowheadOps];
      const attachedArrow = attachedArrowOps[0];
      const target = resolveLeaderTarget(scene, proposal.leader, root, leaderEnd, pageDiagonal);
      attachedArrowOps.forEach((opIndex) => arrowheadOps.add(opIndex));
      const markerSpan = Math.max(0, ...target.markerOps.map((opIndex) => {
        const op = scene.ops[opIndex];
        return op?.kind === "path" ? boundsDiagonal(finiteBounds(op.bounds)) : 0;
      }));
      const targetMergeThreshold = Math.max(
        pageDiagonal * 5e-4,
        cluster.scale * 0.25,
        markerSpan * 0.25
      );
      const semanticTargetTolerance = Math.max(1e-7, pageDiagonal * 1e-6);
      const sameSemanticTarget = (ownedTarget) => pointDistance(ownedTarget, target) <= semanticTargetTolerance || pointDistance(ownedTarget.connection, target.connection) <= targetMergeThreshold;
      const existing = detectedLeaders.find((leader) => leader.targets.some(sameSemanticTarget));
      if (existing) {
        existing.pathOps = [.../* @__PURE__ */ new Set([...existing.pathOps, ...proposal.leader.pathOps])].sort((left, right) => left - right);
        existing.arrowheadOps = [.../* @__PURE__ */ new Set([...existing.arrowheadOps, ...attachedArrowOps])].sort((left, right) => left - right);
        const existingTarget = existing.targets.find(sameSemanticTarget);
        if (existingTarget) {
          existingTarget.markerOps = [.../* @__PURE__ */ new Set([...existingTarget.markerOps, ...target.markerOps])].sort((left, right) => left - right);
          if (existingTarget.arrowheadOp === void 0 && attachedArrow !== void 0) {
            existingTarget.arrowheadOp = attachedArrow;
          }
          const rank = (kind) => kind === "arrow-apex" ? 3 : kind === "open-marker-contact" ? 2 : kind === "marker-contact" ? 1 : 0;
          if (rank(target.terminalKind) > rank(existingTarget.terminalKind)) {
            existingTarget.x = target.x;
            existingTarget.y = target.y;
            existingTarget.terminalKind = target.terminalKind;
          }
        }
        continue;
      }
      detectedLeaders.push({
        pathOps: [...proposal.leader.pathOps],
        arrowheadOps: attachedArrowOps,
        root,
        targets: [{ ...target, arrowheadOp: attachedArrow }]
      });
    }
    const textFrameOps = [];
    const frameSeed = cluster.opIndices[0];
    for (let opIndex = Math.max(0, frameSeed - 2); opIndex <= Math.min(scene.ops.length - 1, frameSeed + 2); opIndex += 1) {
      const op = scene.ops[opIndex];
      if (op.kind === "path" && contains(finiteBounds(op.bounds), cluster.bounds, cluster.scale * 0.6)) {
        textFrameOps.push(opIndex);
      }
    }
    const memberIndices = [...cluster.opIndices, ...textFrameOps, ...owned.flatMap((item) => item.leader.pathOps), ...arrowheadOps];
    const memberOps = memberIndices.map((index) => scene.ops[index]).filter(Boolean);
    const bounds = memberOps.reduce((combined, op) => unionBounds(combined, finiteBounds(op.bounds)), cluster.bounds);
    const groups = [...new Set(memberIndices.map((index) => groupForOp(segmentation, index)))].sort((a, b) => a - b);
    const meanGeometry = owned.reduce((sum, item) => sum + item.geometryScore, 0) / owned.length;
    const meanSequence = owned.reduce((sum, item) => sum + item.sequenceScore, 0) / owned.length;
    const arrowRatio = owned.filter((item) => item.leader.arrowheadOps.length > 0).length / owned.length;
    const conceptualArrowCount = detectedLeaders.filter((leader) => leader.arrowheadOps.length > 0).length;
    const confidence = cluster.source === "vector-outline" ? clamp01(0.05 + 0.72 * meanGeometry + 0.04 * meanSequence + 0.24 * arrowRatio) : clamp01(0.48 * meanGeometry + 0.24 * meanSequence + 0.28 * arrowRatio);
    if (confidence < options.reviewConfidence) continue;
    const sourceRanges = memberOps.map((op) => ({
      startOffset: op.sourceRange.startOffset,
      endOffset: op.sourceRange.endOffset
    })).sort((a, b) => a.startOffset - b.startOffset);
    const memberSet = new Set(memberIndices);
    callouts.push({
      id: `callout-${fnv1a(`${cluster.opIndices.join(",")}|${owned.flatMap((item) => item.leader.pathOps).join(",")}`)}`,
      version: CALLOUT_DETECTION_VERSION,
      status: confidence >= options.formalConfidence ? "formal" : "review",
      subtype: symbolCarrier ? "symbol-callout" : subtypeFor(cluster.text),
      confidence,
      text: cluster.text,
      textOps: [...cluster.opIndices],
      textFrameOps,
      leaders: detectedLeaders,
      sourceRanges,
      bounds,
      primaryGroup: groupForOp(segmentation, cluster.symbolOpIndex ?? cluster.opIndices[0]),
      spannedGroups: groups,
      evidence: {
        rootDistance: Math.min(...owned.map((item) => item.rootDistance)),
        rootThreshold: Math.max(...owned.map((item) => item.rootThreshold)),
        paintGap: Math.min(...owned.map((item) => item.paintGap)),
        sequenceScore: meanSequence,
        geometryScore: meanGeometry,
        arrowheadCount: conceptualArrowCount,
        reasons: [
          symbolCarrier ? cluster.carrierKind === "text-symbol" ? "\u7D27\u90BB\u7684\u77E2\u91CF\u6587\u5B57\u4E0E\u7F16\u53F7\u7B26\u53F7\u5171\u540C\u6784\u6210 callout \u8F7D\u4F53" : "\u95ED\u5408\u7F16\u53F7\u7B26\u53F7\u6784\u6210 callout \u8F7D\u4F53" : cluster.source === "vector-outline" ? "\u8FDE\u7EED\u63CF\u8FB9\u5B57\u5F62\u6784\u6210\u77E2\u91CF\u6587\u5B57\u8F6E\u5ED3\u7C07" : "\u6587\u5B57\u7C07\u5B58\u5728\u552F\u4E00\u5F52\u5C5E\u7684 leader \u6839\u7AEF",
          meanSequence >= 0.5 ? "\u6587\u5B57\u4E0E leader \u7ED8\u5236\u987A\u5E8F\u63A5\u8FD1" : "\u987A\u5E8F\u8F83\u8FDC\uFF0C\u4EE5\u51E0\u4F55\u8FDE\u63A5\u4E3A\u4E3B",
          conceptualArrowCount ? `\u8BC6\u522B\u5230 ${conceptualArrowCount} \u4E2A\u7BAD\u5934\u5934\u90E8` : owned.some((item) => item.leader.packetKind === "open-marker-leader") ? "\u65E0\u7BAD\u5934\u5934\u90E8\uFF0C\u8BC6\u522B\u5230\u5F00\u653E\u5F0F\u7AEF\u70B9\u6807\u8BB0" : "\u65E0\u7BAD\u5934\u5934\u90E8\uFF0C\u6309\u5141\u8BB8\u7684\u65E0\u7BAD\u5934 leader \u5904\u7406",
          ...owned.some((item) => item.leader.spatialEndpointRecovered) ? ["\u6309\u9875\u9762\u4E0E\u6807\u6CE8\u6BD4\u4F8B\u9A8C\u8BC1\u7AEF\u70B9\u63A5\u89E6\uFF0C\u8DE8\u7ED8\u5236\u6279\u6B21\u8865\u5168 leader/\u7BAD\u5934"] : []
        ],
        excludedNearbyOps: spatialIndex.queryBounds(bounds).filter((opIndex) => !memberSet.has(opIndex))
      }
    });
  }
  const diagnosticByGroup = /* @__PURE__ */ new Map();
  for (const leader of leaders) {
    if (claimedLeaders.has(leader.opIndex) || structuralArtifacts.has(leader.opIndex)) continue;
    const group = groupForOp(segmentation, leader.opIndex);
    const existing = diagnosticByGroup.get(group);
    if (existing) {
      existing.count += 1;
      existing.bounds = unionBounds(existing.bounds, finiteBounds(leader.op.bounds));
      if (existing.opIndices.length < 16) existing.opIndices.push(leader.opIndex);
    } else {
      diagnosticByGroup.set(group, {
        kind: "leader-only",
        opIndices: [leader.opIndex],
        count: 1,
        group,
        bounds: finiteBounds(leader.op.bounds),
        message: "\u672C\u7EC4\u5B58\u5728\u7591\u4F3C\u5F15\u7EBF\uFF0C\u4F46\u7AEF\u70B9\u9644\u8FD1\u6CA1\u6709\u53EF\u5173\u8054\u7684\u53EF\u89E3\u7801\u6587\u5B57\u3002"
      });
    }
  }
  const diagnostics = [...diagnosticByGroup.values()].sort((left, right) => left.group - right.group);
  return {
    version: CALLOUT_DETECTION_VERSION,
    options,
    callouts: callouts.sort((a, b) => a.primaryGroup - b.primaryGroup || a.sourceRanges[0].startOffset - b.sourceRanges[0].startOffset),
    diagnostics,
    stats: {
      textClusters: textClusters.length,
      leaderPaths: leaders.length,
      arrowheads: arrowheads.length,
      formal: callouts.filter((callout) => callout.status === "formal").length,
      review: callouts.filter((callout) => callout.status === "review").length
    }
  };
}
function detectCallouts(scene, segmentation, partialOptions = {}) {
  return detectCalloutsInternal(scene, segmentation, partialOptions);
}
function inspectCalloutTextRoi(scene, segmentation, roi, partialOptions = {}) {
  return inspectCalloutTextRois(scene, segmentation, [roi], partialOptions)[0];
}
function inspectCalloutTextRois(scene, segmentation, rois, partialOptions = {}) {
  const normalizedRois = rois.map(finiteBounds);
  if (!normalizedRois.length) return [];
  const spatialIndex = getSceneSpatialIndex(scene);
  const result = detectCalloutsInternal(
    scene,
    segmentation,
    partialOptions,
    normalizedRois
  );
  const calloutsByRoi = normalizedRois.map(() => []);
  for (const callout of result.callouts) {
    const best = normalizedRois.map((roi, index) => ({
      index,
      hitCount: callout.textOps.filter((opIndex) => intersects2(finiteBounds(scene.ops[opIndex].bounds), roi)).length,
      score: 0
    })).map((candidate) => ({
      ...candidate,
      score: candidate.hitCount > 0 ? candidate.hitCount * 2 + candidate.hitCount / Math.max(1, callout.textOps.length) : 0
    })).sort((left, right) => right.score - left.score || left.index - right.index)[0];
    if (best?.score > 0) calloutsByRoi[best.index].push(callout);
  }
  return normalizedRois.map((normalizedRoi, roiIndex) => {
    const callouts = calloutsByRoi[roiIndex];
    const intersectingOps = spatialIndex.queryBounds(normalizedRoi);
    const memberOps = new Set(callouts.flatMap((callout) => [
      ...callout.textOps,
      ...callout.textFrameOps,
      ...callout.leaders.flatMap((leader) => [...leader.pathOps, ...leader.arrowheadOps])
    ]));
    return {
      roi: normalizedRoi,
      intersectingOps,
      ignoredOps: intersectingOps.filter((opIndex) => !memberOps.has(opIndex)),
      callouts
    };
  });
}

// arrow-sidecar/vendor/pdf-engine.ts
var OpStructureBoundary = {
  GraphicsState: 1 << 0,
  TextObject: 1 << 1,
  MarkedContent: 1 << 2,
  CompatibilitySection: 1 << 3
};
var IDENTITY = [1, 0, 0, 1, 0, 0];
var EMPTY_BOUNDS = () => ({
  minX: Number.POSITIVE_INFINITY,
  minY: Number.POSITIVE_INFINITY,
  maxX: Number.NEGATIVE_INFINITY,
  maxY: Number.NEGATIVE_INFINITY
});
var isWhite = (char) => char === "\0" || char === "	" || char === "\n" || char === "\f" || char === "\r" || char === " ";
var isDelimiter = (char) => isWhite(char) || "()<>[]{}/%".includes(char);
var Scanner = class {
  constructor(source) {
    this.source = source;
    this.index = 0;
    this.line = 1;
    this.column = 1;
    this.tokens = 0;
    this.incomplete = false;
    this.issues = [];
  }
  peek(offset = 0) {
    return this.source[this.index + offset] ?? "";
  }
  take() {
    const char = this.source[this.index++] ?? "";
    if (char === "\n") {
      this.line += 1;
      this.column = 1;
    } else {
      this.column += 1;
    }
    return char;
  }
  skipSpaceAndComments() {
    while (this.index < this.source.length) {
      if (isWhite(this.peek())) {
        this.take();
        continue;
      }
      if (this.peek() === "%") {
        while (this.index < this.source.length && this.take() !== "\n") {
        }
        continue;
      }
      break;
    }
  }
  readWord() {
    let value = "";
    while (this.index < this.source.length && !isDelimiter(this.peek())) {
      value += this.take();
    }
    return value;
  }
  readName() {
    this.take();
    let value = "";
    while (this.index < this.source.length && !isDelimiter(this.peek())) {
      if (this.peek() === "#" && /^[0-9a-fA-F]{2}$/.test(`${this.peek(1)}${this.peek(2)}`)) {
        this.take();
        const hex = `${this.take()}${this.take()}`;
        value += String.fromCharCode(Number.parseInt(hex, 16));
      } else {
        value += this.take();
      }
    }
    return { type: "name", value };
  }
  readLiteralString(startLine) {
    this.take();
    let depth = 1;
    let value = "";
    while (this.index < this.source.length) {
      const char = this.take();
      if (char === "\\") {
        if (this.index >= this.source.length) {
          this.incomplete = true;
          return null;
        }
        const escaped = this.take();
        const mapped = {
          n: "\n",
          r: "\r",
          t: "	",
          b: "\b",
          f: "\f",
          "(": "(",
          ")": ")",
          "\\": "\\"
        };
        if (escaped === "\n") continue;
        if (escaped === "\r") {
          if (this.peek() === "\n") this.take();
          continue;
        }
        if (/[0-7]/.test(escaped)) {
          let octal = escaped;
          for (let count = 0; count < 2 && /[0-7]/.test(this.peek()); count += 1) {
            octal += this.take();
          }
          value += String.fromCharCode(Number.parseInt(octal, 8));
        } else {
          value += mapped[escaped] ?? escaped;
        }
      } else if (char === "(") {
        depth += 1;
        value += char;
      } else if (char === ")") {
        depth -= 1;
        if (depth === 0) return { type: "literal", value };
        value += char;
      } else {
        value += char;
      }
    }
    this.incomplete = true;
    this.issues.push({
      level: "warning",
      line: startLine,
      message: "\u8F93\u5165\u672B\u5C3E\u7684\u5B57\u7B26\u4E32\u8FD8\u6CA1\u6709\u53F3\u62EC\u53F7\uFF0C\u5DF2\u7B49\u5F85\u66F4\u591A\u5185\u5BB9\u3002"
    });
    return null;
  }
  readHexString(startLine) {
    this.take();
    let value = "";
    while (this.index < this.source.length && this.peek() !== ">") {
      const char = this.take();
      if (!isWhite(char)) value += char;
    }
    if (this.peek() !== ">") {
      this.incomplete = true;
      this.issues.push({
        level: "warning",
        line: startLine,
        message: "\u8F93\u5165\u672B\u5C3E\u7684\u5341\u516D\u8FDB\u5236\u5B57\u7B26\u4E32\u8FD8\u6CA1\u6709 >\uFF0C\u5DF2\u7B49\u5F85\u66F4\u591A\u5185\u5BB9\u3002"
      });
      return null;
    }
    this.take();
    if (value.length % 2 === 1) value += "0";
    return { type: "hex", value };
  }
  skipDictionary(startLine) {
    this.take();
    this.take();
    let depth = 1;
    while (this.index < this.source.length) {
      if (this.peek() === "(" && !this.readLiteralString(this.line)) return null;
      if (this.peek() === "<" && this.peek(1) === "<") {
        this.take();
        this.take();
        depth += 1;
        continue;
      }
      if (this.peek() === ">" && this.peek(1) === ">") {
        this.take();
        this.take();
        depth -= 1;
        if (depth === 0) return { type: "dict" };
        continue;
      }
      this.take();
    }
    this.incomplete = true;
    this.issues.push({
      level: "warning",
      line: startLine,
      message: "\u8F93\u5165\u672B\u5C3E\u7684\u5B57\u5178\u8FD8\u6CA1\u6709 >>\uFF0C\u5DF2\u7B49\u5F85\u66F4\u591A\u5185\u5BB9\u3002"
    });
    return null;
  }
  readArray(startLine) {
    this.take();
    const value = [];
    while (this.index < this.source.length) {
      this.skipSpaceAndComments();
      if (this.peek() === "]") {
        this.take();
        return { type: "array", value };
      }
      const item = this.readValue(true);
      if (item === void 0) break;
      value.push(item);
    }
    this.incomplete = true;
    this.issues.push({
      level: "warning",
      line: startLine,
      message: "\u8F93\u5165\u672B\u5C3E\u7684\u6570\u7EC4\u8FD8\u6CA1\u6709 ]\uFF0C\u5DF2\u7B49\u5F85\u66F4\u591A\u5185\u5BB9\u3002"
    });
    return null;
  }
  readValue(inArray = false) {
    this.skipSpaceAndComments();
    if (this.index >= this.source.length) return void 0;
    const startLine = this.line;
    const char = this.peek();
    if (")]>{}".includes(char)) {
      this.take();
      return { type: "name", value: char };
    }
    if (char === "/") return this.readName();
    if (char === "(") return this.readLiteralString(startLine) ?? void 0;
    if (char === "[") return this.readArray(startLine) ?? void 0;
    if (char === "<") {
      if (this.peek(1) === "<") return this.skipDictionary(startLine) ?? void 0;
      return this.readHexString(startLine) ?? void 0;
    }
    const word = this.readWord();
    if (word === "true") return true;
    if (word === "false") return false;
    if (word === "null") return null;
    if (/^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(word)) {
      const number = Number(word);
      if (Number.isFinite(number)) return number;
    }
    if (inArray) return { type: "name", value: word };
    return void 0;
  }
  next() {
    this.skipSpaceAndComments();
    if (this.index >= this.source.length) return null;
    const startOffset = this.index;
    const line = this.line;
    const column = this.column;
    const char = this.peek();
    let value;
    if ("/([<".includes(char)) value = this.readValue();
    else if (")]>{}".includes(char)) {
      this.take();
      this.tokens += 1;
      return {
        kind: "keyword",
        value: char,
        line,
        endLine: this.line,
        column,
        startOffset,
        endOffset: this.index
      };
    } else {
      const word = this.readWord();
      if (word === "true") value = true;
      else if (word === "false") value = false;
      else if (word === "null") value = null;
      else if (/^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(word)) {
        const number = Number(word);
        if (Number.isFinite(number)) value = number;
      }
      if (value === void 0) {
        this.tokens += 1;
        return {
          kind: "keyword",
          value: word,
          line,
          endLine: this.line,
          column,
          startOffset,
          endOffset: this.index
        };
      }
    }
    this.tokens += 1;
    if (value === void 0) return null;
    return {
      kind: "value",
      value,
      line,
      endLine: this.line,
      column,
      startOffset,
      endOffset: this.index
    };
  }
  skipInlineImageData() {
    if (isWhite(this.peek())) this.take();
    while (this.index < this.source.length - 3) {
      if (isWhite(this.peek()) && this.peek(1) === "E" && this.peek(2) === "I" && isWhite(this.peek(3))) {
        this.take();
        const startOffset = this.index;
        const startLine = this.line;
        this.take();
        this.take();
        return {
          startOffset,
          endOffset: this.index,
          startLine,
          endLine: this.line
        };
      }
      this.take();
    }
    this.incomplete = true;
    return null;
  }
};
var initialState = () => ({
  ctm: [...IDENTITY],
  lineWidth: 1,
  lineCap: "butt",
  lineJoin: "miter",
  miterLimit: 10,
  dash: [],
  dashPhase: 0,
  strokeColor: "rgb(0 0 0)",
  fillColor: "rgb(0 0 0)",
  strokeAlpha: 1,
  fillAlpha: 1,
  blendMode: "source-over",
  fontName: "sans-serif",
  fontSize: 12,
  charSpacing: 0,
  wordSpacing: 0,
  horizontalScale: 1,
  leading: 0,
  rise: 0,
  renderMode: 0,
  clipPaths: []
});
var cloneBounds = (bounds) => bounds ? { ...bounds } : void 0;
var cloneClipPaths = (paths) => paths.map((path) => ({
  fillRule: path.fillRule,
  segments: path.segments.map((segment) => ({ ...segment }))
}));
var cloneState = (state) => ({
  ...state,
  ctm: [...state.ctm],
  dash: [...state.dash],
  clip: cloneBounds(state.clip),
  clipPaths: cloneClipPaths(state.clipPaths)
});
function multiplyMatrix(left, right) {
  return [
    left[0] * right[0] + left[2] * right[1],
    left[1] * right[0] + left[3] * right[1],
    left[0] * right[2] + left[2] * right[3],
    left[1] * right[2] + left[3] * right[3],
    left[0] * right[4] + left[2] * right[5] + left[4],
    left[1] * right[4] + left[3] * right[5] + left[5]
  ];
}
var transformPoint = (matrix, x, y) => ({
  x: matrix[0] * x + matrix[2] * y + matrix[4],
  y: matrix[1] * x + matrix[3] * y + matrix[5]
});
var matrixScale = (matrix) => {
  const xScale = Math.hypot(matrix[0], matrix[1]);
  const yScale = Math.hypot(matrix[2], matrix[3]);
  return Math.max(1e-6, (xScale + yScale) / 2);
};
var includePoint = (bounds, x, y) => {
  bounds.minX = Math.min(bounds.minX, x);
  bounds.minY = Math.min(bounds.minY, y);
  bounds.maxX = Math.max(bounds.maxX, x);
  bounds.maxY = Math.max(bounds.maxY, y);
};
var isValidBounds = (bounds) => Boolean(
  bounds && Number.isFinite(bounds.minX) && Number.isFinite(bounds.minY) && Number.isFinite(bounds.maxX) && Number.isFinite(bounds.maxY)
);
var unionBounds2 = (left, right) => {
  if (!isValidBounds(left)) return isValidBounds(right) ? { ...right } : void 0;
  if (!isValidBounds(right)) return { ...left };
  return {
    minX: Math.min(left.minX, right.minX),
    minY: Math.min(left.minY, right.minY),
    maxX: Math.max(left.maxX, right.maxX),
    maxY: Math.max(left.maxY, right.maxY)
  };
};
var intersectBounds = (left, right) => {
  if (!left) return right ? { ...right } : void 0;
  if (!right) return { ...left };
  const result = {
    minX: Math.max(left.minX, right.minX),
    minY: Math.max(left.minY, right.minY),
    maxX: Math.min(left.maxX, right.maxX),
    maxY: Math.min(left.maxY, right.maxY)
  };
  return result.minX <= result.maxX && result.minY <= result.maxY ? result : void 0;
};
var expandBounds = (bounds, amount) => ({
  minX: bounds.minX - amount,
  minY: bounds.minY - amount,
  maxX: bounds.maxX + amount,
  maxY: bounds.maxY + amount
});
var rgb = (red, green, blue) => `rgb(${Math.round(Math.max(0, Math.min(1, red)) * 255)} ${Math.round(
  Math.max(0, Math.min(1, green)) * 255
)} ${Math.round(Math.max(0, Math.min(1, blue)) * 255)})`;
var gray = (value) => rgb(value, value, value);
var cmyk = (cyan, magenta, yellow, black) => rgb(
  1 - Math.min(1, cyan + black),
  1 - Math.min(1, magenta + black),
  1 - Math.min(1, yellow + black)
);
var lineCaps = ["butt", "round", "square"];
var lineJoins = ["miter", "round", "bevel"];
var blendModes = {
  Normal: "source-over",
  Compatible: "source-over",
  Multiply: "multiply",
  Screen: "screen",
  Overlay: "overlay",
  Darken: "darken",
  Lighten: "lighten",
  ColorDodge: "color-dodge",
  ColorBurn: "color-burn",
  HardLight: "hard-light",
  SoftLight: "soft-light",
  Difference: "difference",
  Exclusion: "exclusion",
  Hue: "hue",
  Saturation: "saturation",
  Color: "color",
  Luminosity: "luminosity"
};
var asName = (value) => value && typeof value === "object" && "type" in value && value.type === "name" ? value.value : void 0;
var asString = (value) => value && typeof value === "object" && "type" in value && (value.type === "literal" || value.type === "hex") ? value : void 0;
var asArray = (value) => value && typeof value === "object" && "type" in value && value.type === "array" ? value.value : void 0;
var pdfStringBytes = (value) => {
  if (value.type === "literal") {
    return [...value.value].map((char) => char.charCodeAt(0) & 255);
  }
  const bytes = [];
  for (let index = 0; index < value.value.length; index += 2) {
    bytes.push(Number.parseInt(value.value.slice(index, index + 2), 16));
  }
  return bytes;
};
var decodePDFString = (value, font) => {
  const bytes = pdfStringBytes(value);
  if (font && Object.keys(font.toUnicode).length) {
    const lengths = [...new Set(font.codeSpaceLengths.filter((length) => length > 0))].sort((left, right) => right - left);
    const fallbackLength = lengths.at(-1) ?? 1;
    let text = "";
    const codes = [];
    for (let offset = 0; offset < bytes.length; ) {
      let consumed = 0;
      let decoded;
      let code = 0;
      for (const length of lengths) {
        if (offset + length > bytes.length) continue;
        const key = bytes.slice(offset, offset + length).map((byte) => byte.toString(16).padStart(2, "0")).join("").toUpperCase();
        if (font.toUnicode[key] !== void 0) {
          consumed = length;
          decoded = font.toUnicode[key];
          code = Number.parseInt(key, 16);
          break;
        }
      }
      if (!consumed) {
        consumed = Math.min(fallbackLength, bytes.length - offset);
        const key = bytes.slice(offset, offset + consumed).map((byte) => byte.toString(16).padStart(2, "0")).join("").toUpperCase();
        code = Number.parseInt(key || "0", 16);
        decoded = font.toUnicode[key] ?? (code >= 32 && code <= 126 ? String.fromCharCode(code) : "\u25A1");
      }
      codes.push(code);
      text += decoded;
      offset += consumed;
    }
    return { text, codes };
  }
  if (bytes[0] === 254 && bytes[1] === 255) {
    let decoded = "";
    const codes = [];
    for (let index = 2; index + 1 < bytes.length; index += 2) {
      const code = bytes[index] * 256 + bytes[index + 1];
      codes.push(code);
      decoded += String.fromCharCode(code);
    }
    return { text: decoded, codes };
  }
  const isTwoByteCid = bytes.length >= 2 && bytes.length % 2 === 0 && bytes.filter((byte, index) => index % 2 === 0 && byte === 0).length >= bytes.length / 2 - 1;
  if (isTwoByteCid) {
    let decoded = "";
    const codes = [];
    for (let index = 0; index + 1 < bytes.length; index += 2) {
      const cid = bytes[index] * 256 + bytes[index + 1];
      codes.push(cid);
      decoded += cid >= 3 && cid <= 97 ? String.fromCharCode(cid + 29) : "\u25A1";
    }
    return { text: decoded, codes };
  }
  const latin = String.fromCharCode(...bytes);
  const printable = [...latin].filter((char) => {
    const code = char.charCodeAt(0);
    return code === 9 || code === 10 || code >= 32 && code <= 126;
  }).length;
  return {
    text: printable >= latin.length * 0.8 ? latin : "\u25A1".repeat(Math.max(1, bytes.length)),
    codes: bytes
  };
};
function parsePDFStream(source, resources, parseContext = { activeXObjects: /* @__PURE__ */ new Set() }) {
  const started = performance.now();
  const scanner = new Scanner(source);
  const issues = [];
  const issueKeys = /* @__PURE__ */ new Set();
  const unsupported = /* @__PURE__ */ new Set();
  const operands = [];
  let operandStart;
  const ops = [];
  const stateStack = [];
  let state = parseContext.initialState ? cloneState(parseContext.initialState) : initialState();
  const pageResources = parseContext.pageResources ?? resources;
  const completeSource = parseContext.completeSource ?? Boolean(resources);
  let path = [];
  let pathBounds = EMPTY_BOUNDS();
  let pathSourceRange;
  let currentPoint;
  let subpathStart;
  let pendingClip;
  let sceneBounds;
  let pageBounds;
  let textMatrix = [...IDENTITY];
  let lineMatrix = [...IDENTITY];
  let textSourceRange;
  let activeTextOpIndices = [];
  let textObjectSequence = 0;
  let activeTextObjectId;
  let inText = false;
  let operators = 0;
  let paintedPaths = 0;
  let textRuns = 0;
  let nestedTokens = 0;
  let maxGraphicsDepth = 0;
  let inlineImageDictionary = false;
  let pendingStructureClosures = 0;
  let structureBeforeNextPaint = 0;
  const markStructureClose = (flag) => {
    pendingStructureClosures |= flag;
  };
  const markStructureOpen = (flag) => {
    if (pendingStructureClosures & flag) structureBeforeNextPaint |= flag;
  };
  const takeStructureBefore = () => {
    const value = structureBeforeNextPaint;
    structureBeforeNextPaint = 0;
    pendingStructureClosures = 0;
    return value;
  };
  const provenanceFor = (localSourceRange, paintOperator) => ({
    paintOrder: ops.length,
    graphicsDepth: (parseContext.graphicsDepthBase ?? 0) + stateStack.length,
    formInstancePath: [...parseContext.formInstancePath ?? []],
    localSourceRange: { ...localSourceRange },
    localPaintOffset: localSourceRange.endOffset,
    invocationRange: parseContext.pageInvocationRange ? { ...parseContext.pageInvocationRange } : void 0,
    paintOperator,
    textObjectId: activeTextObjectId,
    structureBefore: takeStructureBefore()
  });
  const report = (key, message, line, level = "warning") => {
    if (issueKeys.has(key)) return;
    issueKeys.add(key);
    issues.push({ level, message, line });
  };
  const numbers = (count, operator, line) => {
    const values = operands.slice(-count);
    if (values.length !== count || values.some((value) => typeof value !== "number")) {
      report(
        `arity:${operator}`,
        `${operator} \u7684\u53C2\u6570\u4E0D\u5B8C\u6574\u6216\u7C7B\u578B\u4E0D\u5BF9\uFF1B\u8BE5\u6761\u547D\u4EE4\u5DF2\u8DF3\u8FC7\u3002`,
        line
      );
      return void 0;
    }
    return values;
  };
  const transformed = (x, y) => transformPoint(state.ctm, x, y);
  const extendPathSource = (range) => {
    if (!pathSourceRange) {
      pathSourceRange = { ...range };
      return;
    }
    pathSourceRange.endLine = range.endLine;
    pathSourceRange.endOffset = range.endOffset;
  };
  const addPathPoint = (point) => {
    includePoint(pathBounds, point.x, point.y);
    currentPoint = point;
  };
  const closePath = () => {
    if (!path.length || !subpathStart) return;
    path.push({ kind: "close" });
    currentPoint = subpathStart;
  };
  const applyPendingClip = () => {
    if (!pendingClip || !isValidBounds(pathBounds)) return;
    state.clipPaths.push({
      fillRule: pendingClip,
      segments: path.map((segment) => ({ ...segment }))
    });
    state.clip = intersectBounds(state.clip, pathBounds);
    if (state.clip && !pageBounds) pageBounds = { ...state.clip };
    pendingClip = void 0;
  };
  const clearPath = () => {
    path = [];
    pathBounds = EMPTY_BOUNDS();
    pathSourceRange = void 0;
    currentPoint = void 0;
    subpathStart = void 0;
    pendingClip = void 0;
  };
  const paintPath = (fill, stroke, fillRule, paintRange, paintOperator) => {
    if (path.length && isValidBounds(pathBounds)) {
      const scale = matrixScale(state.ctm);
      const hairline = state.lineWidth === 0;
      const lineWidth = hairline ? 0 : Math.abs(state.lineWidth) * scale;
      const paintedBounds = stroke ? expandBounds(pathBounds, Math.max(lineWidth / 2, 0.25)) : { ...pathBounds };
      const visibleBounds = intersectBounds(paintedBounds, state.clip);
      if (visibleBounds) {
        const sourceRange = pathSourceRange ? {
          ...pathSourceRange,
          endLine: paintRange.endLine,
          endOffset: paintRange.endOffset
        } : { ...paintRange };
        ops.push({
          kind: "path",
          bounds: { ...visibleBounds },
          sourceRange,
          paintOffset: paintRange.endOffset,
          provenance: provenanceFor(sourceRange, paintOperator),
          segments: path.map((segment) => ({ ...segment })),
          fill,
          stroke,
          fillRule,
          fillColor: state.fillColor,
          strokeColor: state.strokeColor,
          fillAlpha: state.fillAlpha,
          strokeAlpha: state.strokeAlpha,
          lineWidth,
          hairline,
          lineCap: state.lineCap,
          lineJoin: state.lineJoin,
          miterLimit: state.miterLimit,
          dash: state.dash.map((dash) => Math.abs(dash) * scale),
          dashPhase: state.dashPhase * scale,
          blendMode: state.blendMode,
          clip: cloneBounds(state.clip),
          clipPaths: cloneClipPaths(state.clipPaths)
        });
        sceneBounds = unionBounds2(sceneBounds, visibleBounds);
        paintedPaths += 1;
      }
    }
    applyPendingClip();
    clearPath();
  };
  const showText = (pdfString, commandRange) => {
    const font = resources?.fonts[state.fontName];
    const decoded = decodePDFString(pdfString, font);
    const safeText = decoded.text.length > 300 ? `${decoded.text.slice(0, 299)}\u2026` : decoded.text;
    if (!safeText) return;
    const matrix = multiplyMatrix(state.ctm, textMatrix);
    const safeCodes = decoded.codes.slice(0, 300);
    const glyphWidth = font ? safeCodes.reduce(
      (total, code) => total + (font.widths[String(code)] ?? font.defaultWidth),
      0
    ) / 1e3 * state.fontSize : Math.max(0.35, safeText.length * 0.56) * state.fontSize;
    const spaces = [...safeText].filter((char) => char === " ").length;
    const glyphAdvance = glyphWidth + Math.max(0, safeCodes.length - 1) * state.charSpacing + spaces * state.wordSpacing;
    const advance = glyphAdvance * state.horizontalScale;
    const baseFont = font?.baseFont?.replace(/^[A-Z]{6}\+/, "") ?? "Arial";
    const fontFamily = /calibri/i.test(baseFont) ? "Calibri" : /arial|helvetica|swiss/i.test(baseFont) ? "Arial" : baseFont.split("-")[0].replace(/[^\w ]/g, " ").trim() || "Arial";
    const fontWeight = /bold|black|demi/i.test(baseFont) ? 700 : 400;
    const localBounds = [
      transformPoint(matrix, 0, state.rise - state.fontSize * 0.25),
      transformPoint(matrix, advance, state.rise - state.fontSize * 0.25),
      transformPoint(matrix, advance, state.rise + state.fontSize * 0.88),
      transformPoint(matrix, 0, state.rise + state.fontSize * 0.88)
    ];
    const bounds = EMPTY_BOUNDS();
    localBounds.forEach((point) => includePoint(bounds, point.x, point.y));
    const visibleBounds = intersectBounds(bounds, state.clip);
    if (state.renderMode !== 3 && visibleBounds) {
      const sourceRange = textSourceRange ? {
        ...textSourceRange,
        endLine: commandRange.endLine,
        endOffset: commandRange.endOffset
      } : { ...commandRange };
      const opIndex = ops.length;
      ops.push({
        kind: "text",
        bounds: { ...visibleBounds },
        sourceRange,
        paintOffset: commandRange.endOffset,
        provenance: provenanceFor(sourceRange, "text"),
        text: safeText,
        matrix,
        fontName: state.fontName,
        fontFamily,
        fontWeight,
        fontSize: Math.max(0.1, Math.abs(state.fontSize)),
        glyphAdvance: Math.max(1e-3, Math.abs(glyphAdvance)),
        horizontalScale: state.horizontalScale,
        rise: state.rise,
        renderMode: state.renderMode,
        fillColor: state.fillColor,
        strokeColor: state.strokeColor,
        fillAlpha: state.fillAlpha,
        strokeAlpha: state.strokeAlpha,
        lineWidth: Math.max(0.25, state.lineWidth),
        blendMode: state.blendMode,
        clip: cloneBounds(state.clip),
        clipPaths: cloneClipPaths(state.clipPaths)
      });
      sceneBounds = unionBounds2(sceneBounds, visibleBounds);
      activeTextOpIndices.push(opIndex);
      textRuns += 1;
    }
    textMatrix = multiplyMatrix(textMatrix, [1, 0, 0, 1, advance, 0]);
  };
  let lexeme;
  while (lexeme = scanner.next()) {
    if (lexeme.kind === "value") {
      if (!operandStart) {
        operandStart = { line: lexeme.line, startOffset: lexeme.startOffset };
      }
      operands.push(lexeme.value);
      continue;
    }
    const operator = lexeme.value;
    const commandRange = {
      startLine: operandStart?.line ?? lexeme.line,
      endLine: lexeme.endLine,
      startOffset: operandStart?.startOffset ?? lexeme.startOffset,
      endOffset: lexeme.endOffset
    };
    operators += 1;
    if (inlineImageDictionary) {
      if (operator === "ID") {
        scanner.skipInlineImageData();
        inlineImageDictionary = false;
        report(
          "inline-image",
          "\u5185\u8054\u56FE\u50CF BI\u2026EI \u6682\u4E0D\u89E3\u7801\uFF0C\u5176\u4ED6\u77E2\u91CF\u547D\u4EE4\u4ECD\u4F1A\u7EE7\u7EED\u7ED8\u5236\u3002",
          lexeme.line
        );
      }
      operands.length = 0;
      operandStart = void 0;
      continue;
    }
    switch (operator) {
      case "q":
        markStructureOpen(OpStructureBoundary.GraphicsState);
        stateStack.push(cloneState(state));
        maxGraphicsDepth = Math.max(maxGraphicsDepth, stateStack.length);
        break;
      case "Q":
        markStructureClose(OpStructureBoundary.GraphicsState);
        if (stateStack.length) state = stateStack.pop();
        else report("q-underflow", "\u9047\u5230\u591A\u4F59\u7684 Q\uFF1B\u5DF2\u5FFD\u7565\u3002", lexeme.line);
        break;
      case "cm": {
        const values = numbers(6, operator, lexeme.line);
        if (values) state.ctm = multiplyMatrix(state.ctm, values);
        break;
      }
      case "w": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.lineWidth = values[0];
        break;
      }
      case "J": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.lineCap = lineCaps[Math.max(0, Math.min(2, values[0] | 0))];
        break;
      }
      case "j": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.lineJoin = lineJoins[Math.max(0, Math.min(2, values[0] | 0))];
        break;
      }
      case "M": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.miterLimit = Math.max(1, values[0]);
        break;
      }
      case "d": {
        const dash = asArray(operands.at(-2));
        const phase = operands.at(-1);
        if (dash && typeof phase === "number" && dash.every((item) => typeof item === "number")) {
          state.dash = dash;
          state.dashPhase = phase;
        } else report("arity:d", "d \u7684\u865A\u7EBF\u6570\u7EC4\u65E0\u6548\uFF1B\u8BE5\u6761\u547D\u4EE4\u5DF2\u8DF3\u8FC7\u3002", lexeme.line);
        break;
      }
      case "m": {
        const values = numbers(2, operator, lexeme.line);
        if (values) {
          extendPathSource(commandRange);
          const point = transformed(values[0], values[1]);
          path.push({ kind: "move", ...point });
          addPathPoint(point);
          subpathStart = point;
        }
        break;
      }
      case "l": {
        const values = numbers(2, operator, lexeme.line);
        if (values) {
          extendPathSource(commandRange);
          const point = transformed(values[0], values[1]);
          if (!currentPoint) {
            path.push({ kind: "move", ...point });
            subpathStart = point;
          } else path.push({ kind: "line", ...point });
          addPathPoint(point);
        }
        break;
      }
      case "c": {
        const values = numbers(6, operator, lexeme.line);
        if (values) {
          extendPathSource(commandRange);
          const first = transformed(values[0], values[1]);
          const second = transformed(values[2], values[3]);
          const end = transformed(values[4], values[5]);
          if (!currentPoint) {
            path.push({ kind: "move", x: first.x, y: first.y });
            subpathStart = first;
          }
          path.push({
            kind: "curve",
            x1: first.x,
            y1: first.y,
            x2: second.x,
            y2: second.y,
            x: end.x,
            y: end.y
          });
          includePoint(pathBounds, first.x, first.y);
          includePoint(pathBounds, second.x, second.y);
          addPathPoint(end);
        }
        break;
      }
      case "v": {
        const values = numbers(4, operator, lexeme.line);
        if (values && currentPoint) {
          extendPathSource(commandRange);
          const second = transformed(values[0], values[1]);
          const end = transformed(values[2], values[3]);
          path.push({
            kind: "curve",
            x1: currentPoint.x,
            y1: currentPoint.y,
            x2: second.x,
            y2: second.y,
            x: end.x,
            y: end.y
          });
          includePoint(pathBounds, second.x, second.y);
          addPathPoint(end);
        }
        break;
      }
      case "y": {
        const values = numbers(4, operator, lexeme.line);
        if (values) {
          extendPathSource(commandRange);
          const first = transformed(values[0], values[1]);
          const end = transformed(values[2], values[3]);
          path.push({
            kind: "curve",
            x1: first.x,
            y1: first.y,
            x2: end.x,
            y2: end.y,
            x: end.x,
            y: end.y
          });
          includePoint(pathBounds, first.x, first.y);
          addPathPoint(end);
        }
        break;
      }
      case "re": {
        const values = numbers(4, operator, lexeme.line);
        if (values) {
          extendPathSource(commandRange);
          const points = [
            transformed(values[0], values[1]),
            transformed(values[0] + values[2], values[1]),
            transformed(values[0] + values[2], values[1] + values[3]),
            transformed(values[0], values[1] + values[3])
          ];
          path.push({ kind: "move", ...points[0] });
          points.slice(1).forEach((point) => path.push({ kind: "line", ...point }));
          path.push({ kind: "close" });
          points.forEach((point) => includePoint(pathBounds, point.x, point.y));
          currentPoint = points[0];
          subpathStart = points[0];
        }
        break;
      }
      case "h":
        if (path.length) {
          extendPathSource(commandRange);
          closePath();
        }
        break;
      case "S":
        paintPath(false, true, "nonzero", commandRange, operator);
        break;
      case "s":
        closePath();
        paintPath(false, true, "nonzero", commandRange, operator);
        break;
      case "f":
      case "F":
        paintPath(true, false, "nonzero", commandRange, operator);
        break;
      case "f*":
        paintPath(true, false, "evenodd", commandRange, operator);
        break;
      case "B":
        paintPath(true, true, "nonzero", commandRange, operator);
        break;
      case "B*":
        paintPath(true, true, "evenodd", commandRange, operator);
        break;
      case "b":
        closePath();
        paintPath(true, true, "nonzero", commandRange, operator);
        break;
      case "b*":
        closePath();
        paintPath(true, true, "evenodd", commandRange, operator);
        break;
      case "n":
        applyPendingClip();
        clearPath();
        break;
      case "W":
        pendingClip = "nonzero";
        break;
      case "W*":
        pendingClip = "evenodd";
        break;
      case "RG": {
        const values = numbers(3, operator, lexeme.line);
        if (values) state.strokeColor = rgb(values[0], values[1], values[2]);
        break;
      }
      case "rg": {
        const values = numbers(3, operator, lexeme.line);
        if (values) state.fillColor = rgb(values[0], values[1], values[2]);
        break;
      }
      case "G": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.strokeColor = gray(values[0]);
        break;
      }
      case "g": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.fillColor = gray(values[0]);
        break;
      }
      case "K": {
        const values = numbers(4, operator, lexeme.line);
        if (values) state.strokeColor = cmyk(values[0], values[1], values[2], values[3]);
        break;
      }
      case "k": {
        const values = numbers(4, operator, lexeme.line);
        if (values) state.fillColor = cmyk(values[0], values[1], values[2], values[3]);
        break;
      }
      case "CS":
      case "cs":
      case "ri":
      case "i":
        break;
      case "BX":
        markStructureOpen(OpStructureBoundary.CompatibilitySection);
        break;
      case "EX":
        markStructureClose(OpStructureBoundary.CompatibilitySection);
        break;
      case "BMC":
      case "BDC":
        markStructureOpen(OpStructureBoundary.MarkedContent);
        break;
      case "EMC":
        markStructureClose(OpStructureBoundary.MarkedContent);
        break;
      case "MP":
      case "DP":
        break;
      case "SC":
      case "SCN": {
        const values = operands.filter((value) => typeof value === "number");
        if (values.length >= 3) state.strokeColor = rgb(values.at(-3), values.at(-2), values.at(-1));
        else if (values.length === 1) state.strokeColor = gray(values[0]);
        break;
      }
      case "sc":
      case "scn": {
        const values = operands.filter((value) => typeof value === "number");
        if (values.length >= 3) state.fillColor = rgb(values.at(-3), values.at(-2), values.at(-1));
        else if (values.length === 1) state.fillColor = gray(values[0]);
        break;
      }
      case "gs": {
        const name = asName(operands.at(-1));
        if (name) {
          const extGState = resources?.extGStates[name];
          if (extGState) {
            if (typeof extGState.fillAlpha === "number") state.fillAlpha = extGState.fillAlpha;
            if (typeof extGState.strokeAlpha === "number") state.strokeAlpha = extGState.strokeAlpha;
            if (extGState.blendMode) {
              const blendMode = blendModes[extGState.blendMode];
              if (blendMode) state.blendMode = blendMode;
              else {
                report(
                  `resource:blend:${extGState.blendMode}`,
                  `\u6DF7\u5408\u6A21\u5F0F /${extGState.blendMode} \u6682\u6309 /Normal \u7ED8\u5236\u3002`,
                  lexeme.line,
                  "info"
                );
              }
            }
          } else {
            const match = name.match(/(\d{1,3})$/);
            if (match) {
              const alpha = Math.max(0, Math.min(1, Number(match[1]) / 255));
              state.fillAlpha = alpha;
              state.strokeAlpha = alpha;
            }
            report(
              `resource:gs:${name}`,
              `/${name} \u7684 ExtGState \u8D44\u6E90\u4E0D\u5728\u5185\u5BB9\u6D41\u4E2D\uFF1B\u900F\u660E\u5EA6\u6309\u540D\u79F0\u8FD1\u4F3C\u3002`,
              lexeme.line,
              "info"
            );
          }
        }
        break;
      }
      case "BT":
        markStructureOpen(OpStructureBoundary.TextObject);
        inText = true;
        textObjectSequence += 1;
        activeTextObjectId = [
          ...parseContext.formInstancePath ?? [],
          `text@${textObjectSequence}`
        ].join("/");
        textMatrix = [...IDENTITY];
        lineMatrix = [...IDENTITY];
        textSourceRange = { ...commandRange };
        activeTextOpIndices = [];
        break;
      case "ET":
        markStructureClose(OpStructureBoundary.TextObject);
        activeTextOpIndices.forEach((index) => {
          ops[index].sourceRange.endLine = commandRange.endLine;
          ops[index].sourceRange.endOffset = commandRange.endOffset;
          if (ops[index].provenance) {
            ops[index].provenance.localSourceRange.endLine = commandRange.endLine;
            ops[index].provenance.localSourceRange.endOffset = commandRange.endOffset;
          }
        });
        inText = false;
        textSourceRange = void 0;
        activeTextOpIndices = [];
        activeTextObjectId = void 0;
        break;
      case "Tf": {
        const name = asName(operands.at(-2));
        const size = operands.at(-1);
        if (name && typeof size === "number") {
          state.fontName = name;
          state.fontSize = size;
          if (!resources?.fonts[name]) {
            report(
              `resource:font:${name}`,
              `/${name} \u7684\u5B57\u4F53\u8D44\u6E90\u7F3A\u5931\uFF0C\u6587\u5B57\u4F1A\u7528\u7CFB\u7EDF\u5B57\u4F53\u8FD1\u4F3C\u3002`,
              lexeme.line,
              "info"
            );
          }
        } else report("arity:Tf", "Tf \u7684\u5B57\u4F53\u540D\u6216\u5B57\u53F7\u65E0\u6548\uFF1B\u5DF2\u8DF3\u8FC7\u3002", lexeme.line);
        break;
      }
      case "Tm": {
        const values = numbers(6, operator, lexeme.line);
        if (values) {
          textMatrix = values;
          lineMatrix = [...textMatrix];
        }
        break;
      }
      case "Td": {
        const values = numbers(2, operator, lexeme.line);
        if (values) {
          lineMatrix = multiplyMatrix(lineMatrix, [1, 0, 0, 1, values[0], values[1]]);
          textMatrix = [...lineMatrix];
        }
        break;
      }
      case "TD": {
        const values = numbers(2, operator, lexeme.line);
        if (values) {
          state.leading = -values[1];
          lineMatrix = multiplyMatrix(lineMatrix, [1, 0, 0, 1, values[0], values[1]]);
          textMatrix = [...lineMatrix];
        }
        break;
      }
      case "T*":
        lineMatrix = multiplyMatrix(lineMatrix, [1, 0, 0, 1, 0, -state.leading]);
        textMatrix = [...lineMatrix];
        break;
      case "Tc": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.charSpacing = values[0];
        break;
      }
      case "Tw": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.wordSpacing = values[0];
        break;
      }
      case "Tz": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.horizontalScale = values[0] / 100;
        break;
      }
      case "TL": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.leading = values[0];
        break;
      }
      case "Ts": {
        const values = numbers(1, operator, lexeme.line);
        if (values) state.rise = values[0];
        break;
      }
      case "Tr": {
        const values = numbers(1, operator, lexeme.line);
        if (values) {
          state.renderMode = Math.max(0, Math.min(7, values[0] | 0));
          if (state.renderMode >= 4) {
            report(
              "text-clipping",
              "\u6587\u5B57\u88C1\u526A\u6A21\u5F0F Tr 4\u20137 \u4F1A\u6309\u53EF\u89C1\u7684\u586B\u5145/\u63CF\u8FB9\u8FD1\u4F3C\u3002",
              lexeme.line,
              "info"
            );
          }
        }
        break;
      }
      case "Tj": {
        const value = asString(operands.at(-1));
        if (value) showText(value, commandRange);
        else report("arity:Tj", "Tj \u7F3A\u5C11\u6587\u5B57\u5B57\u7B26\u4E32\uFF1B\u5DF2\u8DF3\u8FC7\u3002", lexeme.line);
        break;
      }
      case "TJ": {
        const array = asArray(operands.at(-1));
        if (array) {
          array.forEach((item) => {
            const value = asString(item);
            if (value) showText(value, commandRange);
            else if (typeof item === "number") {
              const adjustment = -item / 1e3 * state.fontSize * state.horizontalScale;
              textMatrix = multiplyMatrix(textMatrix, [1, 0, 0, 1, adjustment, 0]);
            }
          });
        } else report("arity:TJ", "TJ \u7F3A\u5C11\u6587\u5B57\u6570\u7EC4\uFF1B\u5DF2\u8DF3\u8FC7\u3002", lexeme.line);
        break;
      }
      case "'": {
        lineMatrix = multiplyMatrix(lineMatrix, [1, 0, 0, 1, 0, -state.leading]);
        textMatrix = [...lineMatrix];
        const value = asString(operands.at(-1));
        if (value) showText(value, commandRange);
        break;
      }
      case '"': {
        const wordSpacing = operands.at(-3);
        const charSpacing = operands.at(-2);
        const value = asString(operands.at(-1));
        if (typeof wordSpacing === "number") state.wordSpacing = wordSpacing;
        if (typeof charSpacing === "number") state.charSpacing = charSpacing;
        lineMatrix = multiplyMatrix(lineMatrix, [1, 0, 0, 1, 0, -state.leading]);
        textMatrix = [...lineMatrix];
        if (value) showText(value, commandRange);
        break;
      }
      case "Do": {
        const name = asName(operands.at(-1)) ?? "XObject";
        const resource = resources?.xObjects?.[name];
        const corners = [
          transformPoint(state.ctm, 0, 0),
          transformPoint(state.ctm, 1, 0),
          transformPoint(state.ctm, 1, 1),
          transformPoint(state.ctm, 0, 1)
        ];
        const bounds = EMPTY_BOUNDS();
        corners.forEach((point) => includePoint(bounds, point.x, point.y));
        const visibleBounds = intersectBounds(bounds, state.clip);
        if (resource?.kind === "image" && visibleBounds) {
          ops.push({
            kind: "image",
            name,
            resourceId: resource.id,
            src: resource.src,
            pixelWidth: resource.width,
            pixelHeight: resource.height,
            softMask: resource.softMask,
            matrix: [...state.ctm],
            corners,
            alpha: state.fillAlpha,
            interpolate: resource.interpolate,
            blendMode: state.blendMode,
            bounds: { ...visibleBounds },
            sourceRange: { ...commandRange },
            paintOffset: commandRange.endOffset,
            provenance: provenanceFor(commandRange, "Do"),
            clip: cloneBounds(state.clip),
            clipPaths: cloneClipPaths(state.clipPaths)
          });
          sceneBounds = unionBounds2(sceneBounds, visibleBounds);
          break;
        }
        if (resource?.kind === "form") {
          const invocationStructureBefore = takeStructureBefore();
          if (resource.transparencyGroup) {
            report(
              `resource:transparency-group:${resource.id}`,
              `/${name} \u662F\u9694\u79BB\u900F\u660E\u5EA6\u7EC4\uFF1B\u5F53\u524D\u4EE5 Canvas \u9010\u56FE\u5143\u6DF7\u5408\u8FD1\u4F3C\u3002`,
              lexeme.line,
              "info"
            );
          }
          if (parseContext.activeXObjects.has(resource.id)) {
            report(
              `resource:xobject-cycle:${resource.id}`,
              `/${name} \u5F62\u6210\u5FAA\u73AF Form XObject \u5F15\u7528\uFF1B\u5DF2\u5728\u6B64\u5904\u505C\u6B62\u9012\u5F52\u3002`,
              lexeme.line
            );
            break;
          }
          const childState = cloneState(state);
          childState.ctm = multiplyMatrix(state.ctm, resource.matrix);
          const formCorners = [
            transformPoint(childState.ctm, resource.bbox.minX, resource.bbox.minY),
            transformPoint(childState.ctm, resource.bbox.maxX, resource.bbox.minY),
            transformPoint(childState.ctm, resource.bbox.maxX, resource.bbox.maxY),
            transformPoint(childState.ctm, resource.bbox.minX, resource.bbox.maxY)
          ];
          const formBounds = EMPTY_BOUNDS();
          formCorners.forEach((point) => includePoint(formBounds, point.x, point.y));
          childState.clip = intersectBounds(state.clip, formBounds);
          childState.clipPaths.push({
            fillRule: "nonzero",
            segments: [
              { kind: "move", ...formCorners[0] },
              ...formCorners.slice(1).map((point) => ({ kind: "line", ...point })),
              { kind: "close" }
            ]
          });
          const activeXObjects = new Set(parseContext.activeXObjects);
          activeXObjects.add(resource.id);
          const formInstancePath = [
            ...parseContext.formInstancePath ?? [],
            `${resource.id}@${commandRange.startOffset}`
          ];
          const child = parsePDFStream(
            resource.source,
            resource.hasOwnResources ? resource.resources : pageResources,
            {
              initialState: childState,
              activeXObjects,
              pageResources,
              completeSource: true,
              formInstancePath,
              pageInvocationRange: parseContext.pageInvocationRange ?? commandRange,
              graphicsDepthBase: (parseContext.graphicsDepthBase ?? 0) + stateStack.length + 1
            }
          );
          child.ops.forEach((op, childIndex) => {
            const provenance = op.provenance && childIndex === 0 ? {
              ...op.provenance,
              structureBefore: (op.provenance.structureBefore ?? 0) | invocationStructureBefore
            } : op.provenance;
            ops.push({
              ...op,
              provenance,
              sourceRange: { ...commandRange },
              paintOffset: commandRange.endOffset
            });
          });
          sceneBounds = unionBounds2(sceneBounds, child.bounds);
          operators += child.stats.operators;
          paintedPaths += child.stats.paintedPaths;
          textRuns += child.stats.textRuns;
          nestedTokens += child.stats.tokens;
          maxGraphicsDepth = Math.max(
            maxGraphicsDepth,
            stateStack.length + child.stats.maxGraphicsDepth + 1
          );
          child.unsupported.forEach((item) => unsupported.add(item));
          child.issues.forEach((issue, index) => {
            report(
              `resource:form:${resource.id}:${index}:${issue.message}`,
              `/${name} \u2192 ${issue.message}`,
              commandRange.startLine,
              issue.level
            );
          });
          break;
        }
        if (visibleBounds && Math.max(
          visibleBounds.maxX - visibleBounds.minX,
          visibleBounds.maxY - visibleBounds.minY
        ) > 0.5) {
          ops.push({
            kind: "placeholder",
            name,
            corners,
            bounds: { ...visibleBounds },
            sourceRange: { ...commandRange },
            paintOffset: commandRange.endOffset,
            provenance: provenanceFor(commandRange, "Do"),
            clip: cloneBounds(state.clip),
            clipPaths: cloneClipPaths(state.clipPaths)
          });
          sceneBounds = unionBounds2(sceneBounds, visibleBounds);
        }
        report(
          `resource:xobject:${name}`,
          resource?.kind === "unsupported" ? `/${name} \u65E0\u6CD5\u7ED8\u5236\uFF1A${resource.reason}` : `/${name} \u7684 XObject \u8D44\u6E90\u7F3A\u5931\uFF1B\u5F53\u524D\u4F4D\u7F6E\u7528\u865A\u7EBF\u6846\u6807\u8BB0\u3002`,
          lexeme.line
        );
        break;
      }
      case "BI":
        inlineImageDictionary = true;
        break;
      case "sh": {
        const name = asName(operands.at(-1)) ?? "Shading";
        report(
          `resource:shading:${name}`,
          `/${name} \u7684\u6E10\u53D8\u8D44\u6E90\u4E0D\u5728\u5185\u5BB9\u6D41\u4E2D\uFF1B\u5DF2\u8DF3\u8FC7\u3002`,
          lexeme.line
        );
        break;
      }
      default:
        if (operator) {
          unsupported.add(operator);
          report(
            `unsupported:${operator}`,
            `\u6682\u4E0D\u652F\u6301\u8FD0\u7B97\u7B26 ${operator}\uFF1B\u540E\u7EED\u547D\u4EE4\u4ECD\u4F1A\u7EE7\u7EED\u6267\u884C\u3002`,
            lexeme.line
          );
        }
    }
    operands.length = 0;
    operandStart = void 0;
  }
  if (!completeSource && path.length && isValidBounds(pathBounds)) {
    const previewRange = pathSourceRange ?? {
      startLine: 1,
      endLine: 1,
      startOffset: 0,
      endOffset: 0
    };
    ops.push({
      kind: "path",
      bounds: { ...intersectBounds(pathBounds, state.clip) ?? pathBounds },
      sourceRange: previewRange,
      paintOffset: pathSourceRange?.endOffset ?? source.length,
      provenance: provenanceFor(previewRange, "preview"),
      segments: path.map((segment) => ({ ...segment })),
      fill: false,
      stroke: true,
      fillRule: "nonzero",
      fillColor: "transparent",
      strokeColor: "rgb(27 199 184)",
      fillAlpha: 0,
      strokeAlpha: 0.9,
      lineWidth: Math.max(0.75, state.lineWidth * matrixScale(state.ctm)),
      hairline: state.lineWidth === 0,
      lineCap: "round",
      lineJoin: "round",
      miterLimit: 10,
      dash: [5 / matrixScale(state.ctm), 4 / matrixScale(state.ctm)],
      dashPhase: 0,
      blendMode: state.blendMode,
      clip: cloneBounds(state.clip),
      clipPaths: cloneClipPaths(state.clipPaths),
      preview: true
    });
    sceneBounds = unionBounds2(sceneBounds, intersectBounds(pathBounds, state.clip));
    report(
      "pending-path",
      "\u6700\u540E\u4E00\u6761\u8DEF\u5F84\u5C1A\u672A\u9047\u5230 S / f / B / n\uFF0C\u5DF2\u7528\u9752\u8272\u865A\u7EBF\u9884\u89C8\u3002",
      void 0,
      "info"
    );
  }
  if (operands.length) {
    report(
      "pending-operands",
      `\u8F93\u5165\u672B\u5C3E\u8FD8\u6709 ${operands.length} \u4E2A\u53C2\u6570\uFF0C\u6B63\u5728\u7B49\u5F85\u8FD0\u7B97\u7B26\u3002`,
      void 0,
      "info"
    );
  }
  if (stateStack.length) {
    report(
      "open-q",
      `\u8FD8\u6709 ${stateStack.length} \u4E2A q \u6CA1\u6709\u5BF9\u5E94\u7684 Q\uFF1B\u5F53\u524D\u7ED3\u679C\u4ECD\u6309\u73B0\u6709\u72B6\u6001\u7ED8\u5236\u3002`,
      void 0,
      "info"
    );
  }
  if (inText) {
    report("open-text", "\u6587\u5B57\u5BF9\u8C61\u8FD8\u6CA1\u6709 ET\uFF1B\u5F53\u524D\u6587\u5B57\u4ECD\u5DF2\u5C1D\u8BD5\u7ED8\u5236\u3002", void 0, "info");
  }
  if (scanner.incomplete) {
    report("incomplete", "\u8F93\u5165\u770B\u8D77\u6765\u5C1A\u672A\u7ED3\u675F\uFF1B\u5DF2\u4FDD\u7559\u5F53\u524D\u53EF\u89E3\u6790\u7ED3\u679C\u3002", void 0, "info");
  }
  issues.push(...scanner.issues);
  ops.forEach((op, paintOrder) => {
    if (op.provenance) op.provenance.paintOrder = paintOrder;
  });
  return {
    ops,
    bounds: sceneBounds,
    pageBounds,
    issues,
    unsupported: [...unsupported],
    stats: {
      operators,
      paintedPaths,
      textRuns,
      tokens: scanner.tokens + nestedTokens,
      maxGraphicsDepth,
      parseMs: performance.now() - started
    }
  };
}

// arrow-sidecar/vendor/callout-page-resource-budget.ts
var MIB = 1024 * 1024;
var DEFAULT_CALLOUT_PAGE_RESOURCE_BUDGET = Object.freeze({
  maxDecodedBytes: 32 * MIB,
  maxSourceLength: 32 * MIB,
  maxSceneOps: 2e5,
  maxPathSegments: 2e6,
  maxSegmentsPerPath: 1e5
});
var CalloutPageResourceLimitError = class extends Error {
  constructor(pageNumber, metric, actual, limit) {
    super(resourceLimitMessage(pageNumber, metric, actual, limit));
    this.pageNumber = pageNumber;
    this.metric = metric;
    this.actual = actual;
    this.limit = limit;
    this.code = "CALLOUT_PAGE_RESOURCE_LIMIT";
    this.name = "CalloutPageResourceLimitError";
  }
};
var finiteLimit = (name, value) => {
  if (!Number.isFinite(value) || value < 1 || !Number.isInteger(value)) {
    throw new TypeError(`\u5355\u9875\u8D44\u6E90\u9884\u7B97 ${name} \u5FC5\u987B\u662F\u5927\u4E8E 0 \u7684\u6709\u9650\u6574\u6570\u3002`);
  }
  return value;
};
var resolveCalloutPageResourceBudget = (partial = {}) => {
  const resolved = { ...DEFAULT_CALLOUT_PAGE_RESOURCE_BUDGET, ...partial };
  return {
    maxDecodedBytes: finiteLimit("maxDecodedBytes", resolved.maxDecodedBytes),
    maxSourceLength: finiteLimit("maxSourceLength", resolved.maxSourceLength),
    maxSceneOps: finiteLimit("maxSceneOps", resolved.maxSceneOps),
    maxPathSegments: finiteLimit("maxPathSegments", resolved.maxPathSegments),
    maxSegmentsPerPath: finiteLimit("maxSegmentsPerPath", resolved.maxSegmentsPerPath)
  };
};
var formatMiB = (value) => `${(value / MIB).toFixed(2)} MiB`;
var pageLabel = (pageNumber) => Number.isInteger(pageNumber) && pageNumber > 0 ? `\u7B2C ${pageNumber} \u9875` : "\u5F53\u524D\u9875";
var resourceLimitMessage = (pageNumber, metric, actual, limit) => {
  const page = pageLabel(pageNumber);
  switch (metric) {
    case "decodedBytes":
      return `${page}\u89E3\u538B\u540E\u7684\u5185\u5BB9\u4E3A ${formatMiB(actual)}\uFF0C\u8D85\u8FC7\u5355\u9875\u9884\u7B97 ${formatMiB(limit)}\u3002\u4E3A\u907F\u514D\u5185\u5B58\u4E0D\u8DB3\uFF0C\u5DF2\u505C\u6B62\u89E3\u6790\uFF1B\u8BF7\u62C6\u5206\u8BE5\u9875\u6216\u63D0\u9AD8 maxDecodedBytes\u3002`;
    case "sourceLength":
      return `${page}\u7684\u77E2\u91CF\u547D\u4EE4\u6E90\u957F\u5EA6\u4E3A ${formatMiB(actual)}\uFF0C\u8D85\u8FC7\u5355\u9875\u9884\u7B97 ${formatMiB(limit)}\u3002\u4E3A\u907F\u514D\u5185\u5B58\u4E0D\u8DB3\uFF0C\u5DF2\u505C\u6B62\u89E3\u6790\uFF1B\u8BF7\u62C6\u5206\u8BE5\u9875\u6216\u63D0\u9AD8 maxSourceLength\u3002`;
    case "sceneOps":
      return `${page}\u89E3\u6790\u51FA ${actual.toLocaleString("zh-CN")} \u4E2A\u7ED8\u5236\u547D\u4EE4\uFF0C\u8D85\u8FC7\u5355\u9875\u9884\u7B97 ${limit.toLocaleString("zh-CN")} \u4E2A\u3002\u8BF7\u62C6\u5206\u8BE5\u9875\u6216\u63D0\u9AD8 maxSceneOps\u3002`;
    case "pathSegments":
      return `${page}\u89E3\u6790\u51FA\u7684\u8DEF\u5F84\u7EBF\u6BB5\u603B\u6570\u4E3A ${actual.toLocaleString("zh-CN")}\uFF0C\u8D85\u8FC7\u5355\u9875\u9884\u7B97 ${limit.toLocaleString("zh-CN")}\uFF08\u5305\u542B\u88C1\u526A\u8DEF\u5F84\uFF09\u3002\u8BF7\u62C6\u5206\u8BE5\u9875\u6216\u63D0\u9AD8 maxPathSegments\u3002`;
    case "segmentsPerPath":
      return `${page}\u5305\u542B\u4E00\u6761\u7531 ${actual.toLocaleString("zh-CN")} \u4E2A\u7EBF\u6BB5\u7EC4\u6210\u7684\u8DEF\u5F84\uFF0C\u8D85\u8FC7\u5355\u8DEF\u5F84\u9884\u7B97 ${limit.toLocaleString("zh-CN")}\u3002\u8BF7\u7B80\u5316\u8BE5\u9875\u77E2\u91CF\u6216\u63D0\u9AD8 maxSegmentsPerPath\u3002`;
  }
};
var assertWithin = (pageNumber, metric, actual, limit) => {
  if (actual > limit) throw new CalloutPageResourceLimitError(pageNumber, metric, actual, limit);
};
var measuredCount = (pageNumber, metric, value) => {
  if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    const label = metric === "decodedBytes" ? "\u89E3\u538B\u5B57\u8282\u6570" : "\u77E2\u91CF\u547D\u4EE4\u6E90\u957F\u5EA6";
    throw new TypeError(`${pageLabel(pageNumber)}\u7684${label}\u7EDF\u8BA1\u65E0\u6548\uFF0C\u65E0\u6CD5\u5B89\u5168\u6267\u884C\u5355\u9875\u89E3\u6790\u3002`);
  }
  return value;
};
function assertPdfPageContentWithinBudget(page, partialBudget = {}) {
  const budget = resolveCalloutPageResourceBudget(partialBudget);
  const seenForms = /* @__PURE__ */ new Set();
  const resourceSourceLength = (resources) => Object.values(resources.xObjects ?? {}).reduce((total, resource) => {
    if (resource.kind !== "form" || seenForms.has(resource)) return total;
    seenForms.add(resource);
    return total + resource.source.length + resourceSourceLength(resource.resources);
  }, 0);
  const usage = {
    decodedBytes: measuredCount(page.pageNumber, "decodedBytes", page.decodedBytes),
    sourceLength: measuredCount(
      page.pageNumber,
      "sourceLength",
      page.source.length + resourceSourceLength(page.resources)
    )
  };
  assertWithin(page.pageNumber, "decodedBytes", usage.decodedBytes, budget.maxDecodedBytes);
  assertWithin(page.pageNumber, "sourceLength", usage.sourceLength, budget.maxSourceLength);
  return usage;
}
function assertSceneWithinCalloutPageBudget(scene, pageNumber, partialBudget = {}) {
  const budget = resolveCalloutPageResourceBudget(partialBudget);
  const usage = {
    sceneOps: scene.ops.length,
    pathSegments: 0,
    largestPathSegments: 0
  };
  assertWithin(pageNumber, "sceneOps", usage.sceneOps, budget.maxSceneOps);
  for (const op of scene.ops) {
    const paths = [
      ...op.kind === "path" ? [op.segments] : [],
      ...op.clipPaths?.map((clipPath) => clipPath.segments) ?? []
    ];
    for (const segments of paths) {
      usage.largestPathSegments = Math.max(usage.largestPathSegments, segments.length);
      assertWithin(
        pageNumber,
        "segmentsPerPath",
        segments.length,
        budget.maxSegmentsPerPath
      );
      usage.pathSegments += segments.length;
      assertWithin(
        pageNumber,
        "pathSegments",
        usage.pathSegments,
        budget.maxPathSegments
      );
    }
  }
  return usage;
}

// arrow-sidecar/vendor/sequential-vector-segmentation.ts
var SEQUENTIAL_SEGMENTATION_VERSION = "sequential-segmentation-v1";
var SequentialStructureBefore = {
  GraphicsState: 1 << 0,
  TextObject: 1 << 1,
  MarkedContent: 1 << 2,
  CompatibilitySection: 1 << 3
};
var DEFAULT_SEQUENTIAL_DISTANCE_THRESHOLDS = Object.freeze({
  spatialJumpRatio: 0.2,
  structuredStyleGapRatio: 0.025,
  structuredGapRatio: 0.05
});
var finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;
var cleanBounds = (bounds) => ({
  minX: finite(Math.min(bounds.minX, bounds.maxX)),
  minY: finite(Math.min(bounds.minY, bounds.maxY)),
  maxX: finite(Math.max(bounds.minX, bounds.maxX)),
  maxY: finite(Math.max(bounds.minY, bounds.maxY))
});
var unionBounds3 = (left, right) => ({
  minX: Math.min(left.minX, right.minX),
  minY: Math.min(left.minY, right.minY),
  maxX: Math.max(left.maxX, right.maxX),
  maxY: Math.max(left.maxY, right.maxY)
});
var boundsGap2 = (left, right) => {
  const dx = Math.max(0, left.minX - right.maxX, right.minX - left.maxX);
  const dy = Math.max(0, left.minY - right.maxY, right.minY - left.maxY);
  return Math.hypot(dx, dy);
};
var boundsDiagonal2 = (bounds) => Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
var sequentialPageDiagonalForBounds = (bounds) => bounds ? Math.max(1, boundsDiagonal2(cleanBounds(bounds))) : 1;
var sequentialPageDiagonalForScene = (scene) => sequentialPageDiagonalForBounds(
  scene.pageBounds ?? scene.bounds ?? scene.ops[0]?.bounds
);
var distanceThresholdsFor = (overrides) => {
  const thresholds = {
    ...DEFAULT_SEQUENTIAL_DISTANCE_THRESHOLDS,
    ...overrides
  };
  for (const [name, value] of Object.entries(thresholds)) {
    if (!Number.isFinite(value) || value < 0) {
      throw new RangeError(`${name} must be a finite non-negative ratio`);
    }
  }
  return thresholds;
};
var lineWidthFor = (op) => op.kind === "path" || op.kind === "text" ? Math.max(0, finite(op.lineWidth)) : 0;
var connectionToleranceFor = (left, right, diagonal) => Math.max(
  5e-4 * diagonal,
  Math.min(5e-3 * diagonal, 2 * Math.max(lineWidthFor(left), lineWidthFor(right)))
);
var GEOMETRY_LEAF_SIZE = 8;
var MAX_CURVE_FLATTENING_DEPTH = 12;
var pointIsFinite = (point) => Number.isFinite(point.x) && Number.isFinite(point.y);
var edgeBounds = (start, end) => ({
  minX: Math.min(start.x, end.x),
  minY: Math.min(start.y, end.y),
  maxX: Math.max(start.x, end.x),
  maxY: Math.max(start.y, end.y)
});
var squaredPointSegmentDistance = (point, start, end) => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= 1e-24) {
    const pointDx2 = point.x - start.x;
    const pointDy2 = point.y - start.y;
    return pointDx2 * pointDx2 + pointDy2 * pointDy2;
  }
  const amount = Math.max(
    0,
    Math.min(
      1,
      ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared
    )
  );
  const nearestX = start.x + amount * dx;
  const nearestY = start.y + amount * dy;
  const pointDx = point.x - nearestX;
  const pointDy = point.y - nearestY;
  return pointDx * pointDx + pointDy * pointDy;
};
var cross = (first, second, third) => (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x);
var pointOnSegment = (point, start, end, epsilon) => Math.abs(cross(start, end, point)) <= epsilon && point.x >= Math.min(start.x, end.x) - epsilon && point.x <= Math.max(start.x, end.x) + epsilon && point.y >= Math.min(start.y, end.y) - epsilon && point.y <= Math.max(start.y, end.y) + epsilon;
var edgesIntersect = (left, right) => {
  const coordinateScale = Math.max(
    1,
    Math.abs(left.start.x),
    Math.abs(left.start.y),
    Math.abs(left.end.x),
    Math.abs(left.end.y),
    Math.abs(right.start.x),
    Math.abs(right.start.y),
    Math.abs(right.end.x),
    Math.abs(right.end.y)
  );
  const epsilon = Number.EPSILON * 64 * coordinateScale * coordinateScale;
  const first = cross(left.start, left.end, right.start);
  const second = cross(left.start, left.end, right.end);
  const third = cross(right.start, right.end, left.start);
  const fourth = cross(right.start, right.end, left.end);
  if ((first > epsilon && second < -epsilon || first < -epsilon && second > epsilon) && (third > epsilon && fourth < -epsilon || third < -epsilon && fourth > epsilon)) return true;
  return Math.abs(first) <= epsilon && pointOnSegment(right.start, left.start, left.end, epsilon) || Math.abs(second) <= epsilon && pointOnSegment(right.end, left.start, left.end, epsilon) || Math.abs(third) <= epsilon && pointOnSegment(left.start, right.start, right.end, epsilon) || Math.abs(fourth) <= epsilon && pointOnSegment(left.end, right.start, right.end, epsilon);
};
var edgeDistance = (left, right) => {
  if (edgesIntersect(left, right)) return 0;
  return Math.sqrt(Math.min(
    squaredPointSegmentDistance(left.start, right.start, right.end),
    squaredPointSegmentDistance(left.end, right.start, right.end),
    squaredPointSegmentDistance(right.start, left.start, left.end),
    squaredPointSegmentDistance(right.end, left.start, left.end)
  ));
};
var midpoint = (left, right) => ({
  x: (left.x + right.x) / 2,
  y: (left.y + right.y) / 2
});
var appendFlattenedCubic = (edges, start, first, second, end, flatnessSquared) => {
  const pending = [{ start, first, second, end, depth: 0 }];
  while (pending.length > 0) {
    const curve = pending.pop();
    const flatness = Math.max(
      squaredPointSegmentDistance(curve.first, curve.start, curve.end),
      squaredPointSegmentDistance(curve.second, curve.start, curve.end)
    );
    if (flatness <= flatnessSquared || curve.depth >= MAX_CURVE_FLATTENING_DEPTH) {
      edges.push({
        start: curve.start,
        end: curve.end,
        bounds: edgeBounds(curve.start, curve.end),
        order: edges.length
      });
      continue;
    }
    const startFirst = midpoint(curve.start, curve.first);
    const firstSecond = midpoint(curve.first, curve.second);
    const secondEnd = midpoint(curve.second, curve.end);
    const leftControl = midpoint(startFirst, firstSecond);
    const rightControl = midpoint(firstSecond, secondEnd);
    const split = midpoint(leftControl, rightControl);
    const depth = curve.depth + 1;
    pending.push({
      start: split,
      first: rightControl,
      second: secondEnd,
      end: curve.end,
      depth
    });
    pending.push({
      start: curve.start,
      first: startFirst,
      second: leftControl,
      end: split,
      depth
    });
  }
};
var spreadMortonBits = (input) => {
  let value = input & 1023;
  value = (value | value << 16) & 50331903;
  value = (value | value << 8) & 50393103;
  value = (value | value << 4) & 51130563;
  value = (value | value << 2) & 153391689;
  return value;
};
var mortonCode = (edge, bounds) => {
  const width = Math.max(1e-12, bounds.maxX - bounds.minX);
  const height = Math.max(1e-12, bounds.maxY - bounds.minY);
  const centerX = (edge.bounds.minX + edge.bounds.maxX) / 2;
  const centerY = (edge.bounds.minY + edge.bounds.maxY) / 2;
  const x = Math.max(0, Math.min(1023, Math.floor(
    (centerX - bounds.minX) / width * 1023
  )));
  const y = Math.max(0, Math.min(1023, Math.floor(
    (centerY - bounds.minY) / height * 1023
  )));
  return (spreadMortonBits(x) | spreadMortonBits(y) << 1) >>> 0;
};
var boundsForEdges = (edges, start, end) => {
  let bounds = { ...edges[start].bounds };
  for (let index = start + 1; index < end; index += 1) {
    bounds = unionBounds3(bounds, edges[index].bounds);
  }
  return bounds;
};
var buildGeometryTree = (edges, start = 0, end = edges.length) => {
  if (start >= end) return void 0;
  const node = {
    bounds: boundsForEdges(edges, start, end),
    start,
    end
  };
  if (end - start <= GEOMETRY_LEAF_SIZE) return node;
  const middle = start + Math.floor((end - start) / 2);
  node.left = buildGeometryTree(edges, start, middle);
  node.right = buildGeometryTree(edges, middle, end);
  return node;
};
var pathGeometryFor = (op, flatness) => {
  const orderedEdges = [];
  let current;
  let subpathStart;
  let subpathHasEdge = false;
  let subpathClosed = false;
  const appendEdge = (start, end) => {
    if (!pointIsFinite(start) || !pointIsFinite(end)) return;
    orderedEdges.push({
      start: { ...start },
      end: { ...end },
      bounds: edgeBounds(start, end),
      order: orderedEdges.length
    });
    subpathHasEdge = true;
  };
  const finishSubpath = () => {
    if (op.fill && subpathHasEdge && !subpathClosed && current && subpathStart) {
      appendEdge(current, subpathStart);
    }
  };
  for (const segment of op.segments) {
    if (segment.kind === "move") {
      finishSubpath();
      current = { x: segment.x, y: segment.y };
      subpathStart = current;
      subpathHasEdge = false;
      subpathClosed = false;
    } else if (segment.kind === "line") {
      const end = { x: segment.x, y: segment.y };
      if (current) appendEdge(current, end);
      else subpathStart = end;
      current = end;
      subpathClosed = false;
    } else if (segment.kind === "curve") {
      const end = { x: segment.x, y: segment.y };
      if (current && pointIsFinite(current) && pointIsFinite(end)) {
        const before = orderedEdges.length;
        appendFlattenedCubic(
          orderedEdges,
          current,
          { x: segment.x1, y: segment.y1 },
          { x: segment.x2, y: segment.y2 },
          end,
          flatness * flatness
        );
        if (orderedEdges.length > before) subpathHasEdge = true;
      } else subpathStart = end;
      current = end;
      subpathClosed = false;
    } else if (current && subpathStart && !subpathClosed) {
      appendEdge(current, subpathStart);
      current = subpathStart;
      subpathClosed = true;
    }
  }
  finishSubpath();
  if (orderedEdges.length === 0) return { edges: [] };
  const first = orderedEdges[0];
  const last = orderedEdges.at(-1);
  const treeEdges = [...orderedEdges];
  const allBounds = boundsForEdges(treeEdges, 0, treeEdges.length);
  treeEdges.sort(
    (left, right) => mortonCode(left, allBounds) - mortonCode(right, allBounds) || left.order - right.order
  );
  return {
    edges: treeEdges,
    root: buildGeometryTree(treeEdges),
    first,
    last
  };
};
var nodeIsLeaf = (node) => !node.left && !node.right;
var pathCenterlineGap = (left, right) => {
  if (!left.root || !right.root || left.edges.length === 0 || right.edges.length === 0) {
    return Number.POSITIVE_INFINITY;
  }
  let best = left.last && right.first ? edgeDistance(left.last, right.first) : Number.POSITIVE_INFINITY;
  const pending = [[left.root, right.root]];
  while (pending.length > 0 && best > 0) {
    const pair = pending.pop();
    const leftNode = pair[0];
    const rightNode = pair[1];
    if (boundsGap2(leftNode.bounds, rightNode.bounds) >= best) continue;
    const leftLeaf = nodeIsLeaf(leftNode);
    const rightLeaf = nodeIsLeaf(rightNode);
    if (leftLeaf && rightLeaf) {
      for (let leftIndex = leftNode.start; leftIndex < leftNode.end; leftIndex += 1) {
        for (let rightIndex = rightNode.start; rightIndex < rightNode.end; rightIndex += 1) {
          best = Math.min(best, edgeDistance(
            left.edges[leftIndex],
            right.edges[rightIndex]
          ));
          if (best === 0) return 0;
        }
      }
      continue;
    }
    const splitLeft = !leftLeaf && (rightLeaf || leftNode.end - leftNode.start >= rightNode.end - rightNode.start);
    const candidates = splitLeft ? [
      [leftNode.left, rightNode],
      [leftNode.right, rightNode]
    ] : [
      [leftNode, rightNode.left],
      [leftNode, rightNode.right]
    ];
    candidates.sort(
      (firstPair, secondPair) => boundsGap2(secondPair[0].bounds, secondPair[1].bounds) - boundsGap2(firstPair[0].bounds, firstPair[1].bounds)
    );
    pending.push(...candidates);
  }
  return best;
};
var visibleStrokeRadius = (op) => {
  if (!op.stroke || op.strokeAlpha <= 0) return 0;
  return op.hairline ? 0.25 : Math.max(0, finite(op.lineWidth)) / 2;
};
var createSequentialGeometryCache = (scene) => {
  const diagonal = sequentialPageDiagonalForScene(scene);
  const curveFlatness = Math.max(1e-7, diagonal * 1e-7);
  const adjacentGaps = new Float64Array(Math.max(0, scene.ops.length - 1));
  adjacentGaps.fill(Number.NaN);
  let pathGeometryCache = [];
  const geometryForOp = (index) => {
    const cachedIndex = pathGeometryCache.findIndex((entry) => entry.index === index);
    if (cachedIndex >= 0) {
      const [cached] = pathGeometryCache.splice(cachedIndex, 1);
      pathGeometryCache.push(cached);
      return cached.geometry ?? void 0;
    }
    const op = scene.ops[index];
    const geometry = op.kind === "path" ? pathGeometryFor(op, curveFlatness) : void 0;
    pathGeometryCache.push({ index, geometry: geometry ?? null });
    if (pathGeometryCache.length > 2) pathGeometryCache.shift();
    return geometry;
  };
  const computeGap = (leftIndex, rightIndex) => {
    const left = scene.ops[leftIndex];
    const right = scene.ops[rightIndex];
    if (!left || !right) return Number.POSITIVE_INFINITY;
    if (left.kind !== "path" || right.kind !== "path") {
      return boundsGap2(cleanBounds(left.bounds), cleanBounds(right.bounds));
    }
    const leftGeometry = geometryForOp(leftIndex);
    const rightGeometry = geometryForOp(rightIndex);
    if (!leftGeometry?.root || !rightGeometry?.root) {
      return boundsGap2(cleanBounds(left.bounds), cleanBounds(right.bounds));
    }
    const centerlineGap = pathCenterlineGap(leftGeometry, rightGeometry);
    return Math.max(
      0,
      centerlineGap - visibleStrokeRadius(left) - visibleStrokeRadius(right)
    );
  };
  return {
    scene,
    diagonal,
    adjacentGap: (leftIndex, rightIndex) => {
      const cacheIndex = rightIndex === leftIndex + 1 ? leftIndex : -1;
      if (cacheIndex >= 0 && cacheIndex < adjacentGaps.length) {
        const cached = adjacentGaps[cacheIndex];
        if (!Number.isNaN(cached)) return cached;
        const gap = computeGap(leftIndex, rightIndex);
        adjacentGaps[cacheIndex] = gap;
        return gap;
      }
      return computeGap(leftIndex, rightIndex);
    },
    retainedGeometryCount: () => pathGeometryCache.length,
    releaseTransientGeometry: () => {
      pathGeometryCache = [];
    }
  };
};
var ratioAtLeast = (left, right, threshold) => {
  const small = Math.min(Math.abs(left), Math.abs(right));
  const large = Math.max(Math.abs(left), Math.abs(right));
  if (small === 0) return large > 0;
  return large / small >= threshold;
};
var sameNumbers = (left, right) => left.length === right.length && left.every((value, index) => value === right[index]);
var uniqueSorted = (values) => [...new Set(values)].sort((left, right) => String(left).localeCompare(String(right)));
var effectivePathColors = (op) => uniqueSorted([
  ...op.fill ? [op.fillColor] : [],
  ...op.stroke ? [op.strokeColor] : []
]);
var effectivePathAlphas = (op) => uniqueSorted([
  ...op.fill ? [op.fillAlpha] : [],
  ...op.stroke ? [op.strokeAlpha] : []
]);
var textChannels = (renderMode) => ({
  fill: renderMode === 0 || renderMode === 2 || renderMode === 4 || renderMode === 6,
  stroke: renderMode === 1 || renderMode === 2 || renderMode === 5 || renderMode === 6
});
var effectiveTextColors = (op) => {
  const channels = textChannels(op.renderMode);
  return uniqueSorted([
    ...channels.fill ? [op.fillColor] : [],
    ...channels.stroke ? [op.strokeColor] : []
  ]);
};
var effectiveTextAlphas = (op) => {
  const channels = textChannels(op.renderMode);
  return uniqueSorted([
    ...channels.fill ? [op.fillAlpha] : [],
    ...channels.stroke ? [op.strokeAlpha] : []
  ]);
};
var colorsDiffer = (left, right) => left.length !== right.length || left.some((value, index) => value !== right[index]);
var alphasDiffer = (left, right) => {
  if (left.length === 0 || right.length === 0) return false;
  const leftMin = Math.min(...left);
  const leftMax = Math.max(...left);
  const rightMin = Math.min(...right);
  const rightMax = Math.max(...right);
  return Math.abs(leftMin - rightMin) > 0.1 || Math.abs(leftMax - rightMax) > 0.1;
};
var strongStyleChanged = (left, right) => {
  if (left.kind !== right.kind) return false;
  if (left.kind === "path" && right.kind === "path") {
    if (colorsDiffer(effectivePathColors(left), effectivePathColors(right))) return true;
    if (alphasDiffer(effectivePathAlphas(left), effectivePathAlphas(right))) return true;
    if (left.blendMode !== right.blendMode) return true;
    if (left.stroke && right.stroke) {
      if (left.hairline !== right.hairline) return true;
      if (ratioAtLeast(left.lineWidth, right.lineWidth, 2)) return true;
      if (!sameNumbers(left.dash, right.dash) || left.dashPhase !== right.dashPhase) return true;
    }
    return false;
  }
  if (left.kind === "text" && right.kind === "text") {
    if (colorsDiffer(effectiveTextColors(left), effectiveTextColors(right))) return true;
    if (alphasDiffer(effectiveTextAlphas(left), effectiveTextAlphas(right))) return true;
    return left.blendMode !== right.blendMode || left.fontName !== right.fontName || ratioAtLeast(left.fontSize, right.fontSize, 1.5);
  }
  if (left.kind === "image" && right.kind === "image") {
    return Math.abs(left.alpha - right.alpha) > 0.1 || left.blendMode !== right.blendMode;
  }
  return false;
};
var contentStreamIdAt = (options, index) => {
  const value = options.contentStreamIds?.[index];
  return value === null || value === void 0 ? "page" : String(value);
};
var sequentialCommandStreamKey = (op, index, options = {}) => {
  const rootFormInstance = op.provenance?.formInstancePath[0] ?? "page";
  return `${contentStreamIdAt(options, index)}${rootFormInstance}`;
};
var structureKinds = (mask) => {
  const result = [];
  if (mask & SequentialStructureBefore.GraphicsState) result.push("graphics-state");
  if (mask & SequentialStructureBefore.TextObject) result.push("text-object");
  if (mask & SequentialStructureBefore.MarkedContent) result.push("marked-content");
  if (mask & SequentialStructureBefore.CompatibilitySection) {
    result.push("compatibility-section");
  }
  return result;
};
var isWhite2 = (char) => char === "\0" || char === "	" || char === "\n" || char === "\f" || char === "\r" || char === " ";
var isDelimiter2 = (char) => char === void 0 || isWhite2(char) || "()<>[]{}/%".includes(char);
var structuralTokens = (source, start, end) => {
  const result = [];
  let index = Math.max(0, Math.min(source.length, start));
  const limit = Math.max(index, Math.min(source.length, end));
  while (index < limit) {
    const char = source[index];
    if (isWhite2(char)) {
      index += 1;
    } else if (char === "%") {
      while (index < limit && source[index] !== "\n" && source[index] !== "\r") index += 1;
    } else if (char === "(") {
      index += 1;
      let depth = 1;
      while (index < limit && depth > 0) {
        if (source[index] === "\\") index += 2;
        else {
          if (source[index] === "(") depth += 1;
          if (source[index] === ")") depth -= 1;
          index += 1;
        }
      }
    } else if (char === "<" && source[index + 1] !== "<") {
      index += 1;
      while (index < limit && source[index] !== ">") index += 1;
      index += 1;
    } else if (char === "/") {
      index += 1;
      while (index < limit && !isDelimiter2(source[index])) index += 1;
    } else if (isDelimiter2(char)) {
      index += char === "<" && source[index + 1] === "<" ? 2 : 1;
    } else {
      const tokenStart = index;
      while (index < limit && !isDelimiter2(source[index])) index += 1;
      const value = source.slice(tokenStart, index);
      if (["Q", "q", "ET", "BT", "EMC", "BDC", "BMC", "EX", "BX"].includes(value)) {
        result.push({ value, start: tokenStart, end: index });
      }
    }
  }
  return result;
};
var orderedStart = (tokens, endValue, startValues) => {
  let sawEnd = false;
  for (const token of tokens) {
    if (token.value === endValue) sawEnd = true;
    else if (sawEnd && startValues.includes(token.value)) return token;
  }
  return void 0;
};
var inspectSource = (source, start, end) => {
  const safeStart = Math.max(0, Math.min(source.length, start));
  const safeEnd = Math.max(safeStart, Math.min(source.length, end));
  const tokens = structuralTokens(source, safeStart, safeEnd);
  let mask = 0;
  if (orderedStart(tokens, "Q", ["q"])) mask |= SequentialStructureBefore.GraphicsState;
  if (orderedStart(tokens, "ET", ["BT"])) mask |= SequentialStructureBefore.TextObject;
  if (orderedStart(tokens, "EMC", ["BDC", "BMC"])) {
    mask |= SequentialStructureBefore.MarkedContent;
  }
  if (orderedStart(tokens, "EX", ["BX"])) {
    mask |= SequentialStructureBefore.CompatibilitySection;
  }
  return {
    mask,
    structures: structureKinds(mask),
    tokens,
    inspectedCharacters: safeEnd - safeStart
  };
};
var formRootFor = (op) => op.provenance?.formInstancePath[0];
var explicitStructureMaskBefore = (right) => {
  const value = right.provenance?.structureBefore;
  return typeof value === "number" && Number.isFinite(value) ? value : void 0;
};
var sourceCutFor = (left, right, inspection, structureMask, sourceLength) => {
  const leftForm = formRootFor(left);
  const rightForm = formRootFor(right);
  if (leftForm !== rightForm) {
    const formOffset = rightForm ? right.sourceRange.startOffset : left.sourceRange.endOffset;
    return Math.max(0, Math.min(sourceLength, formOffset));
  }
  const openingByMask = [
    [SequentialStructureBefore.GraphicsState, ["q"]],
    [SequentialStructureBefore.TextObject, ["BT"]],
    [SequentialStructureBefore.MarkedContent, ["BDC", "BMC"]],
    [SequentialStructureBefore.CompatibilitySection, ["BX"]]
  ];
  const acceptedOpeners = new Set(
    openingByMask.filter(([bit]) => structureMask & bit).flatMap(([, values]) => values)
  );
  const opener = inspection.tokens.find((token) => acceptedOpeners.has(token.value));
  if (opener) return opener.start;
  const endings = /* @__PURE__ */ new Set(["Q", "ET", "EMC", "EX"]);
  const lastEnding = inspection.tokens.findLast((token) => endings.has(token.value));
  if (lastEnding) return lastEnding.end;
  return Math.max(0, Math.min(sourceLength, left.paintOffset));
};
var sourceLengthFor = (scene, source) => source?.length ?? scene.ops.at(-1)?.sourceRange.endOffset ?? 0;
var emptyStats = () => ({
  opCount: 0,
  comparedPairs: 0,
  groupCount: 0,
  boundaryCount: 0,
  inspectedCommandCharacters: 0,
  cutsByReason: {
    "command-stream-switch": 0,
    "spatial-jump": 0,
    "structure-transition": 0
  },
  complexity: "O(N + M + P log P + K) time, O(N + P) space"
});
function segmentSceneSequentially(scene, options = {}) {
  const opCount = scene.ops.length;
  if (options.contentStreamIds && options.contentStreamIds.length !== opCount) {
    throw new RangeError("contentStreamIds must contain one entry per scene operation");
  }
  const assignments = new Int32Array(opCount);
  const boundaries = [];
  const opOffsets = [0];
  const sourceCuts = [0];
  const flatBounds = [];
  const stats = emptyStats();
  stats.opCount = opCount;
  const sourceLength = sourceLengthFor(scene, options.source);
  if (opCount === 0) {
    const hasCommands = sourceLength > 0;
    const fallbackBounds = cleanBounds(
      scene.pageBounds ?? scene.bounds ?? { minX: 0, minY: 0, maxX: 0, maxY: 0 }
    );
    stats.groupCount = hasCommands ? 1 : 0;
    return {
      version: SEQUENTIAL_SEGMENTATION_VERSION,
      assignments,
      groups: {
        count: stats.groupCount,
        opOffsets: hasCommands ? Uint32Array.of(0, 0) : Uint32Array.of(0),
        bounds: hasCommands ? Float64Array.of(
          fallbackBounds.minX,
          fallbackBounds.minY,
          fallbackBounds.maxX,
          fallbackBounds.maxY
        ) : new Float64Array()
      },
      boundaries,
      sourceOffsets: hasCommands ? Uint32Array.of(0, sourceLength) : Uint32Array.of(0),
      stats
    };
  }
  const geometryCache = options.geometryCache?.scene === scene ? options.geometryCache : createSequentialGeometryCache(scene);
  const diagonal = geometryCache.diagonal;
  const distanceThresholds = distanceThresholdsFor(options.distanceThresholds);
  const largeJumpThreshold = Math.max(
    0,
    options.largeJumpDistance ?? distanceThresholds.spatialJumpRatio * diagonal
  );
  const structuredStyleGapThreshold = distanceThresholds.structuredStyleGapRatio * diagonal;
  const structuredGapThreshold = distanceThresholds.structuredGapRatio * diagonal;
  const geometryEpsilon = Math.max(1e-12, diagonal * 1e-9);
  let groupId = 1;
  let groupBounds = cleanBounds(scene.ops[0].bounds);
  assignments[0] = groupId;
  const finishGroup = (nextOpOffset) => {
    opOffsets.push(nextOpOffset);
    flatBounds.push(groupBounds.minX, groupBounds.minY, groupBounds.maxX, groupBounds.maxY);
  };
  for (let index = 1; index < opCount; index += 1) {
    stats.comparedPairs += 1;
    const left = scene.ops[index - 1];
    const right = scene.ops[index];
    const leftStream = sequentialCommandStreamKey(left, index - 1, options);
    const rightStream = sequentialCommandStreamKey(right, index, options);
    const sameFormInstance = formRootFor(left) !== void 0 && formRootFor(left) === formRootFor(right);
    const sourceStart = Math.max(0, Math.min(sourceLength, left.paintOffset));
    const sourceEnd = Math.max(sourceStart, Math.min(sourceLength, right.paintOffset));
    let sourceInspection = {
      mask: 0,
      structures: [],
      tokens: [],
      inspectedCharacters: 0
    };
    const inspectPairSource = () => {
      if (sourceInspection.inspectedCharacters === 0 && options.source && sourceEnd > sourceStart) {
        sourceInspection = inspectSource(options.source, sourceStart, sourceEnd);
        stats.inspectedCommandCharacters += sourceInspection.inspectedCharacters;
      }
      return sourceInspection;
    };
    const explicitStructureMask = explicitStructureMaskBefore(right);
    let structureMask = explicitStructureMask ?? 0;
    let structures = structureKinds(structureMask);
    const strongStyle = strongStyleChanged(left, right);
    const reasons = [];
    if (leftStream !== rightStream) {
      reasons.push({ code: "command-stream-switch", from: leftStream, to: rightStream });
    } else if (!sameFormInstance) {
      const gap = geometryCache.adjacentGap(index - 1, index);
      const connectionTolerance = Math.min(
        connectionToleranceFor(left, right, diagonal),
        largeJumpThreshold
      );
      const connected = gap <= Math.max(geometryEpsilon, connectionTolerance);
      if (explicitStructureMask === void 0) {
        structureMask = inspectPairSource().mask;
        structures = structureKinds(structureMask);
      }
      if (!connected && gap > Math.max(largeJumpThreshold, geometryEpsilon)) {
        reasons.push({ code: "spatial-jump", gap, threshold: largeJumpThreshold });
      }
      const strongStyleBoundary = strongStyle && gap > Math.max(structuredStyleGapThreshold, geometryEpsilon);
      const geometryBoundary = gap > Math.max(structuredGapThreshold, geometryEpsilon);
      if (structures.length > 0 && (strongStyleBoundary || geometryBoundary)) {
        reasons.push({
          code: "structure-transition",
          structures,
          strongStyleChanged: strongStyle,
          gap,
          threshold: strongStyleBoundary ? structuredStyleGapThreshold : structuredGapThreshold
        });
      }
    }
    if (reasons.length > 0) {
      const boundaryIndex = boundaries.length;
      if (options.source && sourceInspection.tokens.length === 0) inspectPairSource();
      const rawCut = sourceCutFor(left, right, sourceInspection, structureMask, sourceLength);
      const sourceOffset = Math.max(sourceCuts.at(-1) ?? 0, Math.min(sourceLength, rawCut));
      boundaries.push({
        index: boundaryIndex,
        beforeOpIndex: index - 1,
        afterOpIndex: index,
        leftGroupId: groupId,
        rightGroupId: groupId + 1,
        sourceOffset,
        reasons,
        structures
      });
      for (const reason of reasons) stats.cutsByReason[reason.code] += 1;
      finishGroup(index);
      sourceCuts.push(sourceOffset);
      groupId += 1;
      groupBounds = cleanBounds(right.bounds);
    } else {
      groupBounds = unionBounds3(groupBounds, cleanBounds(right.bounds));
    }
    assignments[index] = groupId;
  }
  finishGroup(opCount);
  sourceCuts.push(sourceLength);
  stats.groupCount = groupId;
  stats.boundaryCount = boundaries.length;
  return {
    version: SEQUENTIAL_SEGMENTATION_VERSION,
    assignments,
    groups: {
      count: groupId,
      opOffsets: Uint32Array.from(opOffsets),
      bounds: Float64Array.from(flatBounds)
    },
    boundaries,
    sourceOffsets: Uint32Array.from(sourceCuts),
    stats
  };
}

// arrow-sidecar/vendor/callout-page-analysis.ts
var CALLOUT_PAGE_ANALYSIS_VERSION = "callout-page-analysis-v2";
var DEFAULT_CALLOUT_TARGET_REGION_OPTIONS = Object.freeze({
  baseSideShortScale: 8e-3,
  minSideDiagonalScale: 25e-4,
  maxSideDiagonalScale: 6e-3,
  markerPaddingShortScale: 1e-3
});
var finiteBounds2 = (bounds) => ({
  minX: Math.min(bounds.minX, bounds.maxX),
  minY: Math.min(bounds.minY, bounds.maxY),
  maxX: Math.max(bounds.minX, bounds.maxX),
  maxY: Math.max(bounds.minY, bounds.maxY)
});
var optionsEqual = (left, right) => Object.keys(DEFAULT_CALLOUT_DETECTION_OPTIONS).every((key) => left[key] === right[key]);
var detectionReferencesFitScene = (detection, scene) => {
  const last = scene.ops.length - 1;
  return detection.callouts.every((callout) => [
    ...callout.textOps,
    ...callout.textFrameOps,
    ...callout.leaders.flatMap((leader) => [...leader.pathOps, ...leader.arrowheadOps])
  ].every((opIndex) => Number.isInteger(opIndex) && opIndex >= 0 && opIndex <= last));
};
var calloutVectorSignature = (callout) => `${callout.primaryGroup}|${callout.leaders.flatMap((leader) => leader.pathOps).slice().sort((left, right) => left - right).join(",")}`;
var mergeDetectedCallouts = (automatic, inspections) => {
  const merged = new Map(automatic.map((callout) => [calloutVectorSignature(callout), callout]));
  for (const inspection of inspections) {
    for (const callout of inspection.callouts) {
      const signature = calloutVectorSignature(callout);
      if (!merged.has(signature)) merged.set(signature, callout);
    }
  }
  return [...merged.values()].sort((left, right) => left.primaryGroup - right.primaryGroup || (left.sourceRanges[0]?.startOffset ?? 0) - (right.sourceRanges[0]?.startOffset ?? 0));
};
var calloutProvenance = (scene, callouts, automatic, inspections) => {
  const automaticSignatures = new Set(automatic.map(calloutVectorSignature));
  const roiSignatures = new Set(inspections.flatMap((inspection) => inspection.callouts.map(calloutVectorSignature)));
  return callouts.map((callout) => {
    const signature = calloutVectorSignature(callout);
    return {
      calloutId: callout.id,
      textKind: callout.textOps.some((opIndex) => scene.ops[opIndex]?.kind === "text") ? "decoded" : "outline",
      origin: automaticSignatures.has(signature) ? roiSignatures.has(signature) ? "automatic+roi" : "automatic" : "roi"
    };
  });
};
var markerRadiusFrom = (scene, markerOps, center) => markerOps.reduce((radius, opIndex) => {
  const op = scene.ops[opIndex];
  if (!op) return radius;
  const bounds = finiteBounds2(op.bounds);
  return Math.max(
    radius,
    Math.abs(bounds.minX - center.x),
    Math.abs(bounds.maxX - center.x),
    Math.abs(bounds.minY - center.y),
    Math.abs(bounds.maxY - center.y)
  );
}, 0);
function buildCalloutTargetRegions(scene, callouts, pageBounds = finiteBounds2(scene.pageBounds ?? scene.bounds ?? { minX: 0, minY: 0, maxX: 1, maxY: 1 }), partialOptions = {}) {
  const options = { ...DEFAULT_CALLOUT_TARGET_REGION_OPTIONS, ...partialOptions };
  const page = finiteBounds2(pageBounds);
  const width = Math.max(1, page.maxX - page.minX);
  const height = Math.max(1, page.maxY - page.minY);
  const shortSide = Math.min(width, height);
  const diagonal = Math.hypot(width, height);
  const minimum = diagonal * Math.max(0, options.minSideDiagonalScale);
  const maximum = Math.max(minimum, diagonal * Math.max(0, options.maxSideDiagonalScale));
  const baseSide = Math.max(minimum, Math.min(maximum, shortSide * Math.max(0, options.baseSideShortScale)));
  const markerPadding = shortSide * Math.max(0, options.markerPaddingShortScale);
  const regions = [];
  for (const callout of callouts) {
    const calloutTargetCount = callout.leaders.reduce((count, leader) => count + leader.targets.length, 0);
    let ordinal = 0;
    callout.leaders.forEach((leader, leaderIndex) => {
      leader.targets.forEach((target, targetIndex) => {
        ordinal += 1;
        const markerOps = target.markerOps?.length ? [...target.markerOps] : leader.arrowheadOps.length ? [...leader.arrowheadOps] : [];
        const terminalKind = markerOps.length ? "arrowhead" : "free-end";
        const pointKind = target.terminalKind ?? (markerOps.length ? "marker-contact" : "free-end");
        const connection = target.connection ?? { x: target.x, y: target.y };
        const markerRadius = markerRadiusFrom(scene, markerOps, target);
        const halfSide = Math.max(baseSide / 2, markerRadius + (markerOps.length ? markerPadding : 0));
        regions.push({
          id: `${callout.id}:target:${leaderIndex}:${targetIndex}`,
          calloutId: callout.id,
          calloutText: callout.text,
          primaryGroup: callout.primaryGroup,
          leaderIndex,
          targetIndex,
          ordinal,
          calloutTargetCount,
          terminalKind,
          pointKind,
          center: { x: target.x, y: target.y },
          connection: { ...connection },
          bounds: {
            minX: target.x - halfSide,
            minY: target.y - halfSide,
            maxX: target.x + halfSide,
            maxY: target.y + halfSide
          },
          pathOps: [...leader.pathOps],
          markerOps
        });
      });
    });
  }
  return regions;
}
var contentStreamIdsForScene = (scene, ranges) => {
  if (!ranges.length) return void 0;
  const endOffsets = ranges.map((range) => range.endOffset);
  const ids = new Uint32Array(scene.ops.length);
  let streamIndex = 0;
  for (let opIndex = 0; opIndex < scene.ops.length; opIndex += 1) {
    const paintOffset = scene.ops[opIndex].paintOffset;
    while (streamIndex < endOffsets.length && paintOffset > endOffsets[streamIndex]) streamIndex += 1;
    ids[opIndex] = streamIndex;
  }
  return ids;
};
function prepareCalloutPage(page, partialResourceBudget = {}) {
  const resourceBudget = resolveCalloutPageResourceBudget(partialResourceBudget);
  const sourceUsage = assertPdfPageContentWithinBudget(page, resourceBudget);
  const scene = parsePDFStream(page.source, page.resources);
  const sceneUsage = assertSceneWithinCalloutPageBudget(scene, page.pageNumber, resourceBudget);
  const pageBounds = {
    minX: page.boxX,
    minY: page.boxY,
    maxX: page.boxX + page.width,
    maxY: page.boxY + page.height
  };
  scene.pageBounds = pageBounds;
  const contentStreamIds = contentStreamIdsForScene(scene, page.contentStreamRanges);
  const segmentation = segmentSceneSequentially(scene, {
    source: page.source,
    contentStreamIds
  });
  return {
    scene,
    segmentation,
    pageBounds,
    rotation: page.rotation,
    contentStreamIds,
    resourceUsage: { ...sourceUsage, ...sceneUsage }
  };
}
function analyzePreparedCalloutPage(input) {
  const { prepared } = input;
  const requestedOptions = { ...DEFAULT_CALLOUT_DETECTION_OPTIONS, ...input.detectionOptions };
  const cached = input.automaticDetection;
  const automaticDetectionUsed = Boolean(
    cached && cached.version === CALLOUT_DETECTION_VERSION && optionsEqual(cached.options, requestedOptions) && detectionReferencesFitScene(cached, prepared.scene)
  );
  const automaticDetection = automaticDetectionUsed ? cached : detectCallouts(prepared.scene, prepared.segmentation, requestedOptions);
  const roiInspections = inspectCalloutTextRois(
    prepared.scene,
    prepared.segmentation,
    input.textRois ?? [],
    requestedOptions
  );
  const callouts = mergeDetectedCallouts(automaticDetection.callouts, roiInspections);
  const provenance = calloutProvenance(
    prepared.scene,
    callouts,
    automaticDetection.callouts,
    roiInspections
  );
  const targetRegionOptions = {
    ...DEFAULT_CALLOUT_TARGET_REGION_OPTIONS,
    ...input.targetRegionOptions
  };
  const targetRegions = buildCalloutTargetRegions(
    prepared.scene,
    callouts,
    prepared.pageBounds,
    targetRegionOptions
  );
  const detection = {
    ...automaticDetection,
    callouts,
    stats: {
      ...automaticDetection.stats,
      formal: callouts.filter((callout) => callout.status === "formal").length,
      review: callouts.filter((callout) => callout.status === "review").length
    }
  };
  const roiSignatures = new Set(roiInspections.flatMap((inspection) => inspection.callouts.map(calloutVectorSignature)));
  const automaticSignatures = new Set(automaticDetection.callouts.map(calloutVectorSignature));
  return {
    schemaVersion: CALLOUT_PAGE_ANALYSIS_VERSION,
    detectionVersion: CALLOUT_DETECTION_VERSION,
    page: {
      bounds: prepared.pageBounds,
      rotation: prepared.rotation,
      opCount: prepared.scene.ops.length
    },
    groups: {
      count: prepared.segmentation.groups.count,
      opOffsets: Array.from(prepared.segmentation.groups.opOffsets),
      bounds: Array.from(prepared.segmentation.groups.bounds)
    },
    automaticDetection,
    detection,
    roiInspections,
    provenance,
    targetRegionOptions,
    targetRegions,
    stats: {
      automaticCallouts: automaticDetection.callouts.length,
      roiRecoveredCallouts: [...roiSignatures].filter((signature) => !automaticSignatures.has(signature)).length,
      callouts: callouts.length,
      targetRegions: targetRegions.length,
      arrowheadTargets: targetRegions.filter((region) => region.terminalKind === "arrowhead").length,
      freeEndTargets: targetRegions.filter((region) => region.terminalKind === "free-end").length
    },
    cache: { automaticDetectionUsed }
  };
}

// arrow-sidecar/vendor/callout-box-resolver.ts
var overlaps = (left, right) => left.minX <= right.maxX && left.maxX >= right.minX && left.minY <= right.maxY && left.maxY >= right.minY;
var finite2 = (value, fallback) => Number.isFinite(value) ? value : fallback;
var normalizeBox = (box) => {
  const minX = finite2(Math.min(box.minX, box.maxX), 0);
  const maxX = finite2(Math.max(box.minX, box.maxX), 0);
  const minY = finite2(Math.min(box.minY, box.maxY), 0);
  const maxY = finite2(Math.max(box.minY, box.maxY), 0);
  return { minX, minY, maxX, maxY };
};
var unionBounds4 = (scene, opIndices, fallback) => {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const opIndex of opIndices) {
    const bounds = scene.ops[opIndex]?.bounds;
    if (!bounds) continue;
    minX = Math.min(minX, bounds.minX);
    minY = Math.min(minY, bounds.minY);
    maxX = Math.max(maxX, bounds.maxX);
    maxY = Math.max(maxY, bounds.maxY);
  }
  return Number.isFinite(minX) ? { minX, minY, maxX, maxY } : fallback;
};
var boxId = (box) => {
  const key = `${box.minX.toFixed(3)},${box.minY.toFixed(3)},${box.maxX.toFixed(3)},${box.maxY.toFixed(3)}`;
  let hash = 2166136261;
  for (let index = 0; index < key.length; index += 1) {
    hash ^= key.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return `callout-box-${hash.toString(16).padStart(8, "0")}`;
};
var matchAutomaticCallout = (scene, callouts, box) => {
  let best = null;
  for (const callout of callouts) {
    const primary = callout.textOps.filter((opIndex) => Boolean(scene.ops[opIndex]));
    const carrier = primary.length ? primary : callout.textFrameOps.filter((opIndex) => Boolean(scene.ops[opIndex]));
    if (!carrier.length) continue;
    const hits = carrier.filter((opIndex) => overlaps(scene.ops[opIndex].bounds, box));
    if (!hits.length) continue;
    const score = hits.length * 4 + hits.length / carrier.length + callout.confidence;
    if (!best || score > best.score) best = { callout, hits, score };
  }
  return best ? { callout: best.callout, hits: best.hits } : null;
};
var synthesizeBoxCallout = (scene, segmentation, box, intersectingOps) => {
  const textOps = intersectingOps.filter((opIndex) => scene.ops[opIndex]?.kind === "text");
  const carrierOps = textOps.length ? textOps : [...intersectingOps];
  const text = textOps.map((opIndex) => {
    const op = scene.ops[opIndex];
    return op && op.kind === "text" ? op.text : "";
  }).join(" ").replace(/\s+/g, " ").trim();
  const assignments = segmentation?.assignments;
  const groups = /* @__PURE__ */ new Set();
  for (const opIndex of carrierOps) {
    const group = assignments?.[opIndex];
    if (typeof group === "number" && group >= 0) groups.add(group);
  }
  const spannedGroups = [...groups].sort((left, right) => left - right);
  const sourceRanges = carrierOps.map((opIndex) => scene.ops[opIndex]?.sourceRange).filter((range) => Boolean(range)).map((range) => ({ startOffset: range.startOffset, endOffset: range.endOffset })).sort((left, right) => left.startOffset - right.startOffset);
  return {
    id: boxId(box),
    version: "callout-box-resolver-v1",
    // Nothing was geometrically confirmed here, so this must not claim the
    // same standing as a detected callout.
    status: "review",
    subtype: "note",
    confidence: 0,
    text,
    textOps,
    textFrameOps: textOps.length ? [] : carrierOps,
    leaders: [],
    sourceRanges,
    bounds: unionBounds4(scene, carrierOps, normalizeBox(box)),
    primaryGroup: spannedGroups[0] ?? -1,
    spannedGroups,
    evidence: {
      rootDistance: 0,
      rootThreshold: 0,
      paintGap: 0,
      sequenceScore: 0,
      geometryScore: 0,
      arrowheadCount: 0,
      reasons: ["\u6846\u5185\u672A\u627E\u5230\u5F15\u7EBF\u6216\u7BAD\u5934\uFF0C\u6309\u8C03\u7528\u65B9\u65AD\u8A00\u4FDD\u7559 callout \u8F7D\u4F53"],
      excludedNearbyOps: []
    }
  };
};
function resolveCalloutBox(prepared, box, options = {}) {
  const { scene, segmentation, pageBounds } = prepared;
  const roi = normalizeBox(box);
  const detectionOptions = options.detectionOptions ?? {};
  const finish = (source, callout, intersectingOps2) => {
    const targetRegions = callout.leaders.length ? buildCalloutTargetRegions(scene, [callout], pageBounds, options.targetRegionOptions) : [];
    return {
      roi,
      source,
      callout,
      targetRegions,
      hasLeader: callout.leaders.length > 0,
      intersectingOps: intersectingOps2
    };
  };
  const automatic = options.automaticCallouts?.length ? matchAutomaticCallout(scene, options.automaticCallouts, roi) : null;
  if (automatic) return finish("automatic", automatic.callout, automatic.hits);
  const inspection = inspectCalloutTextRoi(scene, segmentation, roi, detectionOptions);
  const intersectingOps = inspection?.intersectingOps ?? [];
  const roiCallout = inspection?.callouts?.length ? [...inspection.callouts].sort((left, right) => right.leaders.length - left.leaders.length || right.confidence - left.confidence)[0] : null;
  if (roiCallout) return finish("roi", roiCallout, intersectingOps);
  return finish(
    "text-only",
    synthesizeBoxCallout(scene, segmentation, roi, intersectingOps),
    intersectingOps
  );
}
function resolveCalloutBoxes(prepared, boxes, options = {}) {
  return boxes.map((box) => resolveCalloutBox(prepared, box, options));
}

// arrow-sidecar/vendor/pdf-file.ts
var DEFAULT_PDF_PAGE_EXTRACTION_BUDGET = Object.freeze({
  maxDecodedBytes: 32 * 1024 * 1024,
  maxDecodedStreamBytes: 16 * 1024 * 1024
});
var pdfPageExtractionResourceLimitError = (message) => Object.assign(
  new Error(message),
  { code: "CALLOUT_PAGE_RESOURCE_LIMIT" }
);
var byteString = (bytes) => {
  let output = "";
  const chunkSize = 32768;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    output += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return output;
};
var normalizedRotation = (angle) => {
  const normalized = (angle % 360 + 360) % 360;
  return normalized === 90 || normalized === 180 || normalized === 270 ? normalized : 0;
};
var bytesToHex = (bytes) => {
  let output = "";
  for (const value of bytes) output += value.toString(16).padStart(2, "0");
  return output;
};
var fingerprintPdfBytes = async (bytes) => {
  try {
    const subtle = globalThis.crypto?.subtle;
    if (subtle) {
      const digest = await subtle.digest("SHA-256", bytes);
      return `sha256:${bytesToHex(new Uint8Array(digest))}`;
    }
  } catch {
  }
  let first = 2166136261;
  let second = 2654435769;
  for (let index = 0; index < bytes.length; index += 1) {
    const value = bytes[index];
    first = Math.imul(first ^ value, 16777619) >>> 0;
    second = Math.imul(second ^ value + index, 2246822507) >>> 0;
  }
  return `fallback:${bytes.length.toString(36)}:${first.toString(16)}${second.toString(16)}`;
};
async function loadPdfBytes(input, options = {}) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const header = byteString(bytes.subarray(0, Math.min(bytes.length, 1024)));
  if (!header.includes("%PDF-")) {
    throw new Error("\u6587\u4EF6\u5934\u4E2D\u6CA1\u6709\u627E\u5230 %PDF- \u6807\u8BB0\uFF0C\u8BF7\u786E\u8BA4\u8FD9\u662F\u6709\u6548\u7684 PDF \u6587\u4EF6\u3002");
  }
  const fingerprintPromise = fingerprintPdfBytes(bytes);
  const { PDFDocument } = await import("pdf-lib");
  const document2 = await PDFDocument.load(bytes, {
    ignoreEncryption: true,
    throwOnInvalidObject: true,
    updateMetadata: false
  });
  if (document2.isEncrypted) {
    throw new Error("\u5F53\u524D PDF \u5DF2\u52A0\u5BC6\u3002\u6B64\u7248\u672C\u6682\u4E0D\u652F\u6301\u9700\u8981\u5BC6\u7801\u6216\u5E26\u6743\u9650\u52A0\u5BC6\u7684 PDF\u3002");
  }
  const pageCount = document2.getPageCount();
  if (!pageCount) throw new Error("PDF \u4E2D\u6CA1\u6709\u53EF\u89E3\u6790\u7684\u9875\u9762\u3002");
  return {
    document: document2,
    fileName: options.fileName ?? "document.pdf",
    fileSize: options.fileSize ?? bytes.byteLength,
    fingerprint: await fingerprintPromise,
    pageCount,
    title: document2.getTitle()?.trim() || void 0
  };
}
var decodeStream = async (stream, pdfLib) => {
  if (stream instanceof pdfLib.PDFRawStream) {
    return pdfLib.decodePDFRawStream(stream).decode();
  }
  const generatedStream = stream;
  if (typeof generatedStream.getUnencodedContents === "function") {
    return generatedStream.getUnencodedContents();
  }
  return stream.getContents();
};
var unicodeFromHex = (hex) => {
  const normalized = hex.length % 2 ? `${hex}0` : hex;
  const bytes = [];
  for (let index = 0; index < normalized.length; index += 2) {
    bytes.push(Number.parseInt(normalized.slice(index, index + 2), 16));
  }
  const start = bytes[0] === 254 && bytes[1] === 255 ? 2 : 0;
  let output = "";
  for (let index = start; index + 1 < bytes.length; index += 2) {
    output += String.fromCharCode(bytes[index] * 256 + bytes[index + 1]);
  }
  return output;
};
var incrementHex = (hex, amount) => {
  const incremented = (BigInt(`0x${hex}`) + BigInt(amount)).toString(16);
  return incremented.padStart(hex.length, "0").toUpperCase();
};
var parseToUnicodeCMap = (source) => {
  const toUnicode = {};
  const codeSpaceLengths = /* @__PURE__ */ new Set();
  for (const block of source.matchAll(/begincodespacerange([\s\S]*?)endcodespacerange/gi)) {
    for (const pair of block[1].matchAll(/<([0-9a-f]+)>\s*<([0-9a-f]+)>/gi)) {
      codeSpaceLengths.add(Math.max(1, Math.ceil(pair[1].length / 2)));
    }
  }
  for (const block of source.matchAll(/beginbfchar([\s\S]*?)endbfchar/gi)) {
    for (const pair of block[1].matchAll(/<([0-9a-f]+)>\s*<([0-9a-f]+)>/gi)) {
      const key = pair[1].toUpperCase();
      toUnicode[key] = unicodeFromHex(pair[2]);
      codeSpaceLengths.add(Math.max(1, Math.ceil(key.length / 2)));
    }
  }
  for (const block of source.matchAll(/beginbfrange([\s\S]*?)endbfrange/gi)) {
    const ranges = block[1].matchAll(
      /<([0-9a-f]+)>\s*<([0-9a-f]+)>\s*(?:<([0-9a-f]+)>|\[([\s\S]*?)\])/gi
    );
    for (const range of ranges) {
      const startHex = range[1].toUpperCase();
      const endHex = range[2].toUpperCase();
      const start = Number.parseInt(startHex, 16);
      const end = Number.parseInt(endHex, 16);
      const count = Math.min(65536, Math.max(0, end - start + 1));
      const destinations = range[4] ? [...range[4].matchAll(/<([0-9a-f]+)>/gi)].map((item) => item[1]) : [];
      for (let offset = 0; offset < count; offset += 1) {
        const key = (start + offset).toString(16).padStart(startHex.length, "0").toUpperCase();
        const destination = range[3] ? incrementHex(range[3], offset) : destinations[offset];
        if (destination) toUnicode[key] = unicodeFromHex(destination);
      }
      codeSpaceLengths.add(Math.max(1, Math.ceil(startHex.length / 2)));
    }
  }
  return {
    toUnicode,
    codeSpaceLengths: [...codeSpaceLengths].sort((left, right) => left - right)
  };
};
var numberFromDictionary = (dictionary, key, pdfLib) => dictionary.lookupMaybe(pdfLib.PDFName.of(key), pdfLib.PDFNumber)?.asNumber();
var numberFromArray = (array, index, pdfLib) => array.lookupMaybe(index, pdfLib.PDFNumber)?.asNumber();
var extractFontWidths = (font, pdfLib) => {
  const widths = {};
  const descendants = font.lookupMaybe(pdfLib.PDFName.of("DescendantFonts"), pdfLib.PDFArray);
  const descendant = descendants?.lookupMaybe(0, pdfLib.PDFDict);
  if (descendant) {
    const widthArray2 = descendant.lookupMaybe(pdfLib.PDFName.of("W"), pdfLib.PDFArray);
    const defaultWidth = numberFromDictionary(descendant, "DW", pdfLib) ?? 1e3;
    if (widthArray2) {
      for (let index = 0; index < widthArray2.size(); ) {
        const firstCode = numberFromArray(widthArray2, index, pdfLib);
        if (firstCode === void 0) break;
        const widthListObject = widthArray2.lookup(index + 1);
        if (widthListObject instanceof pdfLib.PDFArray) {
          const widthList = widthListObject;
          for (let offset = 0; offset < widthList.size(); offset += 1) {
            const width2 = numberFromArray(widthList, offset, pdfLib);
            if (width2 !== void 0) widths[String(firstCode + offset)] = width2;
          }
          index += 2;
          continue;
        }
        const lastCode = numberFromArray(widthArray2, index + 1, pdfLib);
        const width = numberFromArray(widthArray2, index + 2, pdfLib);
        if (lastCode === void 0 || width === void 0) break;
        for (let code = firstCode; code <= lastCode && code - firstCode < 65536; code += 1) {
          widths[String(code)] = width;
        }
        index += 3;
      }
    }
    return { widths, defaultWidth };
  }
  const firstChar = numberFromDictionary(font, "FirstChar", pdfLib) ?? 0;
  const widthArray = font.lookupMaybe(pdfLib.PDFName.of("Widths"), pdfLib.PDFArray);
  if (widthArray) {
    for (let index = 0; index < widthArray.size(); index += 1) {
      const width = numberFromArray(widthArray, index, pdfLib);
      if (width !== void 0) widths[String(firstChar + index)] = width;
    }
  }
  const descriptor = font.lookupMaybe(pdfLib.PDFName.of("FontDescriptor"), pdfLib.PDFDict);
  return {
    widths,
    defaultWidth: descriptor ? numberFromDictionary(descriptor, "MissingWidth", pdfLib) ?? 500 : 500
  };
};
var resolvedFromDictionary = (dictionary, key, pdfLib) => {
  const value = dictionary.get(pdfLib.PDFName.of(key));
  return value ? dictionary.context.lookup(value) : void 0;
};
var numbersFromPdfArray = (array, count, pdfLib) => {
  if (!array || array.size() < count) return void 0;
  const values = Array.from(
    { length: count },
    (_, index) => numberFromArray(array, index, pdfLib)
  );
  return values.every((value) => value !== void 0) ? values : void 0;
};
var matrixFromDictionary = (dictionary, pdfLib) => {
  const array = dictionary.lookupMaybe(pdfLib.PDFName.of("Matrix"), pdfLib.PDFArray);
  const values = numbersFromPdfArray(array, 6, pdfLib);
  return values ?? [1, 0, 0, 1, 0, 0];
};
var boundsFromDictionary = (dictionary, pdfLib) => {
  const array = dictionary.lookupMaybe(pdfLib.PDFName.of("BBox"), pdfLib.PDFArray);
  const values = numbersFromPdfArray(array, 4, pdfLib) ?? [0, 0, 1, 1];
  return {
    minX: Math.min(values[0], values[2]),
    minY: Math.min(values[1], values[3]),
    maxX: Math.max(values[0], values[2]),
    maxY: Math.max(values[1], values[3])
  };
};
var dataUrl = (mimeType, bytes) => `data:${mimeType};base64,${btoa(byteString(bytes))}`;
var crcTable = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = value & 1 ? 3988292384 ^ value >>> 1 : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();
var crc32 = (bytes) => {
  let value = 4294967295;
  for (const byte of bytes) value = crcTable[(value ^ byte) & 255] ^ value >>> 8;
  return (value ^ 4294967295) >>> 0;
};
var concatenateBytes = (...parts) => {
  const output = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
};
var uint32Bytes = (value) => new Uint8Array([
  value >>> 24 & 255,
  value >>> 16 & 255,
  value >>> 8 & 255,
  value & 255
]);
var pngChunk = (name, contents) => {
  const type = new TextEncoder().encode(name);
  return concatenateBytes(
    uint32Bytes(contents.length),
    type,
    contents,
    uint32Bytes(crc32(concatenateBytes(type, contents)))
  );
};
var pngDataUrl = async (width, height, colorType, scanlines, options = {}) => {
  const header = new Uint8Array(13);
  header.set(uint32Bytes(width), 0);
  header.set(uint32Bytes(height), 4);
  header[8] = options.bitDepth ?? 8;
  header[9] = colorType;
  const compressedStream = new Blob([Uint8Array.from(scanlines)]).stream().pipeThrough(new CompressionStream("deflate"));
  const compressed = new Uint8Array(await new Response(compressedStream).arrayBuffer());
  const png = concatenateBytes(
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", header),
    ...options.palette ? [pngChunk("PLTE", options.palette)] : [],
    pngChunk("IDAT", compressed),
    pngChunk("IEND", new Uint8Array())
  );
  return dataUrl("image/png", png);
};
var filterNames = (stream, pdfLib) => {
  const filter = resolvedFromDictionary(stream.dict, "Filter", pdfLib);
  if (filter instanceof pdfLib.PDFName) return [filter.decodeText()];
  if (filter instanceof pdfLib.PDFArray) {
    const names = [];
    for (let index = 0; index < filter.size(); index += 1) {
      const item = filter.lookup(index);
      if (item instanceof pdfLib.PDFName) names.push(item.decodeText());
    }
    return names;
  }
  return [];
};
var MAX_IMAGE_PIXELS = 2e7;
var MAX_PAGE_IMAGE_PIXELS = 8e7;
var throwIfPdfPageAborted = (signal) => {
  if (signal?.aborted) throw signal.reason ?? new DOMException("PDF page extraction aborted.", "AbortError");
};
var decompressDeflate = async (bytes) => {
  const stream = new Blob([Uint8Array.from(bytes)]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
};
var decodeAsciiHex = (bytes) => {
  const source = byteString(bytes).replace(/\s+/g, "").replace(/>.*/, "");
  const padded = source.length % 2 ? `${source}0` : source;
  const output = new Uint8Array(padded.length / 2);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Number.parseInt(padded.slice(index * 2, index * 2 + 2), 16);
  }
  return output;
};
var decodeAscii85 = (bytes) => {
  const source = byteString(bytes).replace(/\s+/g, "").replace(/^<~/, "").replace(/~>.*/, "");
  const output = [];
  let group = [];
  const flush = (partial = false) => {
    if (!group.length) return;
    const originalLength = group.length;
    while (group.length < 5) group.push(84);
    let value = 0;
    for (const digit of group) value = value * 85 + digit;
    const decoded = [value >>> 24, value >>> 16, value >>> 8, value].map((item) => item & 255);
    output.push(...decoded.slice(0, partial ? Math.max(0, originalLength - 1) : 4));
    group = [];
  };
  for (const char of source) {
    if (char === "z" && group.length === 0) {
      output.push(0, 0, 0, 0);
      continue;
    }
    const digit = char.charCodeAt(0) - 33;
    if (digit < 0 || digit > 84) continue;
    group.push(digit);
    if (group.length === 5) flush(false);
  }
  flush(true);
  return Uint8Array.from(output);
};
var unwrapTerminalImage = async (encoded, filters, terminal) => {
  let bytes = encoded;
  const terminalIndex = filters.indexOf(terminal);
  for (const filter of filters.slice(0, terminalIndex)) {
    if (filter === "FlateDecode" || filter === "Fl") bytes = await decompressDeflate(bytes);
    else if (filter === "ASCII85Decode" || filter === "A85") bytes = decodeAscii85(bytes);
    else if (filter === "ASCIIHexDecode" || filter === "AHx") bytes = decodeAsciiHex(bytes);
    else throw new Error(`\u6682\u4E0D\u652F\u6301 ${terminal} \u524D\u7684\u5916\u5C42\u6EE4\u955C /${filter}\u3002`);
  }
  return bytes;
};
var extractImageXObject = async (stream, id, pdfLib, budget, allowMasks = true) => {
  const width = numberFromDictionary(stream.dict, "Width", pdfLib) ?? 0;
  const height = numberFromDictionary(stream.dict, "Height", pdfLib) ?? 0;
  const bits = numberFromDictionary(stream.dict, "BitsPerComponent", pdfLib) ?? 8;
  const interpolate = Boolean(
    stream.dict.lookupMaybe(pdfLib.PDFName.of("Interpolate"), pdfLib.PDFBool)?.asBoolean()
  );
  if (width <= 0 || height <= 0) {
    return { kind: "unsupported", id, reason: "\u56FE\u50CF\u5BBD\u9AD8\u65E0\u6548\u3002" };
  }
  const pixels = width * height;
  const firstVisit = !budget.seenImages.has(stream);
  if (pixels > MAX_IMAGE_PIXELS || firstVisit && budget.imagePixels + pixels > MAX_PAGE_IMAGE_PIXELS) {
    return {
      kind: "unsupported",
      id,
      reason: `\u56FE\u50CF\u50CF\u7D20\u91CF ${pixels.toLocaleString()} \u8D85\u8FC7\u5B89\u5168\u89E3\u6790\u9884\u7B97\u3002`
    };
  }
  if (firstVisit) {
    budget.seenImages.add(stream);
    budget.imagePixels += pixels;
  }
  const attachTransparency = async (image) => {
    if (!allowMasks) return image;
    const explicitMask = resolvedFromDictionary(stream.dict, "Mask", pdfLib);
    if (explicitMask && !(explicitMask instanceof pdfLib.PDFName && explicitMask.decodeText() === "None")) {
      return {
        kind: "unsupported",
        id,
        reason: "\u56FE\u50CF\u4F7F\u7528\u4E86\u663E\u5F0F /Mask\uFF1B\u4E3A\u907F\u514D\u9519\u8BEF\u7684\u4E0D\u900F\u660E\u5E95\u8272\uFF0C\u5F53\u524D\u672A\u7ED8\u5236\u3002"
      };
    }
    const softMaskObject = resolvedFromDictionary(stream.dict, "SMask", pdfLib);
    if (!softMaskObject || softMaskObject instanceof pdfLib.PDFName && softMaskObject.decodeText() === "None") {
      return image;
    }
    if (!(softMaskObject instanceof pdfLib.PDFStream)) {
      return { kind: "unsupported", id, reason: "\u56FE\u50CF /SMask \u4E0D\u662F\u53EF\u89E3\u7801\u7684\u56FE\u50CF\u6D41\u3002" };
    }
    const softMask = await extractImageXObject(
      softMaskObject,
      `${id}:smask`,
      pdfLib,
      budget,
      false
    );
    if (softMask.kind !== "image") {
      return {
        kind: "unsupported",
        id,
        reason: `\u56FE\u50CF\u8F6F\u906E\u7F69\u65E0\u6CD5\u89E3\u7801\uFF1A${softMask.kind === "unsupported" ? softMask.reason : "\u8F6F\u906E\u7F69\u6D41\u4E0D\u662F\u56FE\u50CF"}`
      };
    }
    const decode = numbersFromPdfArray(
      softMaskObject.dict.lookupMaybe(pdfLib.PDFName.of("Decode"), pdfLib.PDFArray),
      2,
      pdfLib
    );
    return {
      ...image,
      softMask: {
        src: softMask.src,
        width: softMask.width,
        height: softMask.height,
        invert: Boolean(decode && decode[0] > decode[1])
      }
    };
  };
  const filters = filterNames(stream, pdfLib);
  const encoded = stream.getContents();
  if (filters.includes("DCTDecode")) {
    try {
      const jpeg = await unwrapTerminalImage(encoded, filters, "DCTDecode");
      if (jpeg[0] !== 255 || jpeg[1] !== 216) throw new Error("\u89E3\u7801\u540E\u6CA1\u6709\u627E\u5230 JPEG SOI \u6807\u8BB0\u3002");
      return attachTransparency({
        kind: "image",
        id,
        width,
        height,
        src: dataUrl("image/jpeg", jpeg),
        interpolate
      });
    } catch (error) {
      return {
        kind: "unsupported",
        id,
        reason: error instanceof Error ? error.message : "JPEG \u5916\u5C42\u6EE4\u955C\u89E3\u7801\u5931\u8D25\u3002"
      };
    }
  }
  if (filters.includes("JPXDecode")) {
    return {
      kind: "unsupported",
      id,
      reason: "JPEG 2000 /JPXDecode \u5728\u6D4F\u89C8\u5668\u4E0E\u9A8C\u8BC1\u6E32\u67D3\u5668\u4E2D\u90FD\u6CA1\u6709\u7A33\u5B9A\u89E3\u7801\u652F\u6301\u3002"
    };
  }
  const colorSpace = resolvedFromDictionary(stream.dict, "ColorSpace", pdfLib);
  const colorSpaceName = colorSpace instanceof pdfLib.PDFName ? colorSpace.decodeText() : colorSpace instanceof pdfLib.PDFArray && colorSpace.lookup(0) instanceof pdfLib.PDFName ? colorSpace.lookup(0).decodeText() : "";
  let components = colorSpaceName === "DeviceGray" ? 1 : colorSpaceName === "DeviceRGB" ? 3 : 0;
  let pngColorType = components === 1 ? 0 : 2;
  let palette;
  if (colorSpaceName === "Indexed" && colorSpace instanceof pdfLib.PDFArray) {
    const base = colorSpace.lookup(1);
    const highValue = colorSpace.lookup(2);
    const lookup = colorSpace.lookup(3);
    const baseName = base instanceof pdfLib.PDFName ? base.decodeText() : "";
    const paletteEntries = highValue instanceof pdfLib.PDFNumber ? Math.max(1, Math.min(256, highValue.asNumber() + 1)) : 0;
    let lookupBytes;
    if (lookup instanceof pdfLib.PDFString || lookup instanceof pdfLib.PDFHexString) {
      lookupBytes = lookup.asBytes();
    } else if (lookup instanceof pdfLib.PDFStream) {
      lookupBytes = await decodeStream(lookup, pdfLib);
    }
    const baseComponents = baseName === "DeviceGray" ? 1 : baseName === "DeviceRGB" ? 3 : 0;
    if (paletteEntries && lookupBytes && baseComponents) {
      palette = new Uint8Array(paletteEntries * 3);
      for (let entry = 0; entry < paletteEntries; entry += 1) {
        if (baseComponents === 1) {
          const gray2 = lookupBytes[entry] ?? 0;
          palette.set([gray2, gray2, gray2], entry * 3);
        } else {
          palette.set(lookupBytes.subarray(entry * 3, entry * 3 + 3), entry * 3);
        }
      }
      components = 1;
      pngColorType = 3;
    }
  }
  if (!components) {
    return {
      kind: "unsupported",
      id,
      reason: `\u6682\u4E0D\u652F\u6301\u56FE\u50CF\u989C\u8272\u7A7A\u95F4 ${colorSpace?.toString() ?? "(missing)"}\u3002`
    };
  }
  if (![1, 2, 4, 8].includes(bits) || pngColorType !== 3 && bits !== 8) {
    return { kind: "unsupported", id, reason: `\u6682\u4E0D\u652F\u6301 ${bits} \u4F4D ${colorSpaceName} \u56FE\u50CF\u91C7\u6837\u3002` };
  }
  let decoded;
  try {
    decoded = await decodeStream(stream, pdfLib);
  } catch (error) {
    throwIfPdfPageAborted(budget.signal);
    return {
      kind: "unsupported",
      id,
      reason: error instanceof Error ? error.message : "\u56FE\u50CF\u6EE4\u955C\u89E3\u7801\u5931\u8D25\u3002"
    };
  }
  const rowBytes = Math.ceil(width * components * bits / 8);
  const decodeParameters = resolvedFromDictionary(stream.dict, "DecodeParms", pdfLib);
  const parameterDictionaries = [];
  if (decodeParameters instanceof pdfLib.PDFDict) parameterDictionaries.push(decodeParameters);
  if (decodeParameters instanceof pdfLib.PDFArray) {
    for (let index = 0; index < decodeParameters.size(); index += 1) {
      const item = decodeParameters.lookup(index);
      if (item instanceof pdfLib.PDFDict) parameterDictionaries.push(item);
    }
  }
  const predictor = parameterDictionaries.map((dictionary) => numberFromDictionary(dictionary, "Predictor", pdfLib)).find((value) => value !== void 0) ?? 1;
  let scanlines;
  if (decoded.length === (rowBytes + 1) * height) {
    for (let row = 0; row < height; row += 1) {
      if (decoded[row * (rowBytes + 1)] > 4) {
        return { kind: "unsupported", id, reason: "PNG Predictor \u884C\u6EE4\u955C\u503C\u8D85\u51FA 0\u20134\u3002" };
      }
    }
    scanlines = decoded;
  } else if (decoded.length >= rowBytes * height && decoded.length <= rowBytes * height + 16) {
    decoded = decoded.subarray(0, rowBytes * height);
    if (predictor === 2) {
      if (bits !== 8) {
        return { kind: "unsupported", id, reason: "\u6682\u4E0D\u652F\u6301\u6253\u5305\u91C7\u6837\u7684 TIFF Predictor 2\u3002" };
      }
      const restored = Uint8Array.from(decoded);
      for (let row = 0; row < height; row += 1) {
        const rowStart = row * rowBytes;
        for (let offset = components; offset < rowBytes; offset += 1) {
          restored[rowStart + offset] = restored[rowStart + offset] + restored[rowStart + offset - components] & 255;
        }
      }
      decoded = restored;
    } else if (predictor >= 10) {
      return {
        kind: "unsupported",
        id,
        reason: `PNG Predictor ${predictor} \u7F3A\u5C11\u6BCF\u884C\u6EE4\u955C\u5B57\u8282\u3002`
      };
    }
    scanlines = new Uint8Array((rowBytes + 1) * height);
    for (let row = 0; row < height; row += 1) {
      scanlines.set(decoded.subarray(row * rowBytes, (row + 1) * rowBytes), row * (rowBytes + 1) + 1);
    }
  } else {
    return {
      kind: "unsupported",
      id,
      reason: `\u89E3\u7801\u540E\u50CF\u7D20\u957F\u5EA6 ${decoded.length} \u4E0E ${width}\xD7${height} \u4E0D\u5339\u914D\u3002`
    };
  }
  return attachTransparency({
    kind: "image",
    id,
    width,
    height,
    src: await pngDataUrl(width, height, pngColorType, scanlines, {
      bitDepth: bits,
      palette
    }),
    interpolate
  });
};
var resourceIdentifier = (value, fallback, pdfLib) => value instanceof pdfLib.PDFRef ? value.toString() : fallback;
var extractXObjectResource = async (stream, id, pdfLib, active, depth, scope, budget) => {
  throwIfPdfPageAborted(budget.signal);
  const subtype = stream.dict.lookupMaybe(
    pdfLib.PDFName.of("Subtype"),
    pdfLib.PDFName
  )?.decodeText();
  if (subtype === "Image") return extractImageXObject(stream, id, pdfLib, budget);
  if (subtype !== "Form") {
    return { kind: "unsupported", id, reason: `\u4E0D\u652F\u6301\u7684 XObject subtype /${subtype ?? "Unknown"}\u3002` };
  }
  if (depth >= 16 || active.has(id)) {
    return { kind: "unsupported", id, reason: "Form XObject \u5FAA\u73AF\u5F15\u7528\u6216\u5D4C\u5957\u8D85\u8FC7 16 \u5C42\u3002" };
  }
  const nextActive = new Set(active);
  nextActive.add(id);
  const localResources = stream.dict.lookupMaybe(
    pdfLib.PDFName.of("Resources"),
    pdfLib.PDFDict
  );
  const group = stream.dict.lookupMaybe(
    pdfLib.PDFName.of("Group"),
    pdfLib.PDFDict
  );
  const groupSubtype = group?.lookupMaybe(
    pdfLib.PDFName.of("S"),
    pdfLib.PDFName
  )?.decodeText();
  return {
    kind: "form",
    id,
    source: byteString(await decodeStream(stream, pdfLib)).replace(/\r\n?/g, "\n"),
    matrix: matrixFromDictionary(stream.dict, pdfLib),
    bbox: boundsFromDictionary(stream.dict, pdfLib),
    hasOwnResources: Boolean(localResources),
    transparencyGroup: groupSubtype === "Transparency",
    resources: await extractPageResources(
      localResources,
      pdfLib,
      nextActive,
      depth + 1,
      scope,
      budget
    )
  };
};
async function extractPageResources(resourceDictionary, pdfLib, active = /* @__PURE__ */ new Set(), depth = 0, scope = "page", budget = { imagePixels: 0, seenImages: /* @__PURE__ */ new WeakSet() }) {
  const resources = { fonts: {}, extGStates: {}, xObjects: {} };
  if (!resourceDictionary) return resources;
  const fonts = resourceDictionary.lookupMaybe(pdfLib.PDFName.of("Font"), pdfLib.PDFDict);
  for (const [fontName, fontObject] of fonts?.entries() ?? []) {
    throwIfPdfPageAborted(budget.signal);
    const name = fontName.decodeText();
    const font = fonts.context.lookup(fontObject, pdfLib.PDFDict);
    const baseFont = font.lookupMaybe(pdfLib.PDFName.of("BaseFont"), pdfLib.PDFName)?.decodeText();
    const cmapStream = font.lookupMaybe(pdfLib.PDFName.of("ToUnicode"), pdfLib.PDFStream);
    let cmap = {
      toUnicode: {},
      codeSpaceLengths: [1]
    };
    if (cmapStream) {
      try {
        cmap = parseToUnicodeCMap(byteString(await decodeStream(cmapStream, pdfLib)));
      } catch {
        throwIfPdfPageAborted(budget.signal);
      }
    }
    resources.fonts[name] = {
      baseFont,
      ...cmap,
      ...extractFontWidths(font, pdfLib)
    };
  }
  const extGStates = resourceDictionary.lookupMaybe(
    pdfLib.PDFName.of("ExtGState"),
    pdfLib.PDFDict
  );
  for (const [stateName, stateObject] of extGStates?.entries() ?? []) {
    throwIfPdfPageAborted(budget.signal);
    const state = extGStates.context.lookup(stateObject, pdfLib.PDFDict);
    const blendValue = resolvedFromDictionary(state, "BM", pdfLib);
    const blendMode = blendValue instanceof pdfLib.PDFName ? blendValue.decodeText() : blendValue instanceof pdfLib.PDFArray && blendValue.lookup(0) instanceof pdfLib.PDFName ? blendValue.lookup(0).decodeText() : void 0;
    resources.extGStates[stateName.decodeText()] = {
      strokeAlpha: numberFromDictionary(state, "CA", pdfLib),
      fillAlpha: numberFromDictionary(state, "ca", pdfLib),
      blendMode
    };
  }
  const xObjects = resourceDictionary.lookupMaybe(
    pdfLib.PDFName.of("XObject"),
    pdfLib.PDFDict
  );
  for (const [objectName, objectValue] of xObjects?.entries() ?? []) {
    throwIfPdfPageAborted(budget.signal);
    const name = objectName.decodeText();
    const id = resourceIdentifier(objectValue, `direct:${scope}:${name}`, pdfLib);
    try {
      const stream = xObjects.context.lookup(objectValue, pdfLib.PDFStream);
      resources.xObjects[name] = await extractXObjectResource(
        stream,
        id,
        pdfLib,
        active,
        depth,
        `${scope}/${name}`,
        budget
      );
    } catch (error) {
      throwIfPdfPageAborted(budget.signal);
      resources.xObjects[name] = {
        kind: "unsupported",
        id,
        reason: error instanceof Error ? error.message : "XObject \u89E3\u6790\u5931\u8D25\u3002"
      };
    }
  }
  return resources;
}
var multiplyPdfMatrix = (left, right) => [
  left[0] * right[0] + left[2] * right[1],
  left[1] * right[0] + left[3] * right[1],
  left[0] * right[2] + left[2] * right[3],
  left[1] * right[2] + left[3] * right[3],
  left[0] * right[4] + left[2] * right[5] + left[4],
  left[1] * right[4] + left[3] * right[5] + left[5]
];
var transformedBounds = (matrix, bounds) => {
  const points = [
    [bounds.minX, bounds.minY],
    [bounds.maxX, bounds.minY],
    [bounds.maxX, bounds.maxY],
    [bounds.minX, bounds.maxY]
  ].map(([x, y]) => ({
    x: matrix[0] * x + matrix[2] * y + matrix[4],
    y: matrix[1] * x + matrix[3] * y + matrix[5]
  }));
  return {
    minX: Math.min(...points.map((point) => point.x)),
    minY: Math.min(...points.map((point) => point.y)),
    maxX: Math.max(...points.map((point) => point.x)),
    maxY: Math.max(...points.map((point) => point.y))
  };
};
var annotationAppearanceStream = (annotation, pdfLib) => {
  const appearance = annotation.lookupMaybe(pdfLib.PDFName.of("AP"), pdfLib.PDFDict);
  if (!appearance) return void 0;
  const normal = resolvedFromDictionary(appearance, "N", pdfLib);
  if (normal instanceof pdfLib.PDFStream) return normal;
  if (!(normal instanceof pdfLib.PDFDict)) return void 0;
  const stateName = annotation.lookupMaybe(pdfLib.PDFName.of("AS"), pdfLib.PDFName);
  const stateValue = stateName ? normal.get(stateName) : normal.get(pdfLib.PDFName.of("Off"));
  if (!stateValue) return void 0;
  const resolved = normal.context.lookup(stateValue);
  return resolved instanceof pdfLib.PDFStream ? resolved : void 0;
};
var extractAnnotationAppearances = async (annotations, resources, pdfLib, budget) => {
  const commands = [];
  let decodedBytes = 0;
  let count = 0;
  for (let index = 0; index < (annotations?.size() ?? 0); index += 1) {
    throwIfPdfPageAborted(budget.signal);
    const annotation = annotations.lookupMaybe(index, pdfLib.PDFDict);
    if (!annotation) continue;
    const flags = numberFromDictionary(annotation, "F", pdfLib) ?? 0;
    if ((flags & (1 | 2 | 32)) !== 0) continue;
    const stream = annotationAppearanceStream(annotation, pdfLib);
    if (!stream) continue;
    const rawAnnotation = annotations.get(index);
    const id = `${resourceIdentifier(rawAnnotation, `direct:annotation:${index}`, pdfLib)}:appearance`;
    const resource = await extractXObjectResource(
      stream,
      id,
      pdfLib,
      /* @__PURE__ */ new Set(),
      0,
      `annotation:${index + 1}`,
      budget
    );
    if (resource.kind !== "form") continue;
    const rectangle = numbersFromPdfArray(
      annotation.lookupMaybe(pdfLib.PDFName.of("Rect"), pdfLib.PDFArray),
      4,
      pdfLib
    );
    if (rectangle) {
      const rectBounds = {
        minX: Math.min(rectangle[0], rectangle[2]),
        minY: Math.min(rectangle[1], rectangle[3]),
        maxX: Math.max(rectangle[0], rectangle[2]),
        maxY: Math.max(rectangle[1], rectangle[3])
      };
      const paintedBounds = transformedBounds(resource.matrix, resource.bbox);
      const paintedWidth = Math.max(1e-6, paintedBounds.maxX - paintedBounds.minX);
      const paintedHeight = Math.max(1e-6, paintedBounds.maxY - paintedBounds.minY);
      const scaleX = (rectBounds.maxX - rectBounds.minX) / paintedWidth;
      const scaleY = (rectBounds.maxY - rectBounds.minY) / paintedHeight;
      const placement = [
        scaleX,
        0,
        0,
        scaleY,
        rectBounds.minX - paintedBounds.minX * scaleX,
        rectBounds.minY - paintedBounds.minY * scaleY
      ];
      resource.matrix = multiplyPdfMatrix(placement, resource.matrix);
    }
    const name = `__Annotation${index + 1}`;
    resources.xObjects ??= {};
    resources.xObjects[name] = resource;
    const subtype = annotation.lookupMaybe(
      pdfLib.PDFName.of("Subtype"),
      pdfLib.PDFName
    )?.decodeText();
    commands.push(`% Annotation ${index + 1}${subtype ? ` /${subtype}` : ""}
q
/${name} Do
Q`);
    decodedBytes += resource.source.length;
    count += 1;
  }
  return { commands, count, decodedBytes };
};
async function extractPdfPage(loaded, pageNumber, options = {}) {
  throwIfPdfPageAborted(options.signal);
  if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > loaded.pageCount) {
    throw new Error(`\u9875\u7801\u5FC5\u987B\u5728 1 \u5230 ${loaded.pageCount} \u4E4B\u95F4\u3002`);
  }
  const pdfLib = await import("pdf-lib");
  const page = loaded.document.getPage(pageNumber - 1);
  const contents = page.node.Contents();
  const streams = [];
  if (contents instanceof pdfLib.PDFArray) {
    for (let index = 0; index < contents.size(); index += 1) {
      const stream = contents.lookupMaybe(index, pdfLib.PDFStream);
      if (!stream) {
        throw new Error(`\u7B2C ${pageNumber} \u9875\u7684\u7B2C ${index + 1} \u4E2A Contents \u9879\u4E0D\u662F\u5185\u5BB9\u6D41\u3002`);
      }
      streams.push(stream);
    }
  } else if (contents) {
    streams.push(contents);
  }
  const budget = {
    ...DEFAULT_PDF_PAGE_EXTRACTION_BUDGET,
    ...options.maxDecodedBytes === void 0 ? {} : { maxDecodedBytes: options.maxDecodedBytes },
    ...options.maxDecodedStreamBytes === void 0 ? {} : { maxDecodedStreamBytes: options.maxDecodedStreamBytes }
  };
  if (!Number.isSafeInteger(budget.maxDecodedBytes) || budget.maxDecodedBytes < 1 || !Number.isSafeInteger(budget.maxDecodedStreamBytes) || budget.maxDecodedStreamBytes < 1) {
    throw new TypeError("PDF \u5355\u9875\u89E3\u538B\u9884\u7B97\u5FC5\u987B\u662F\u5927\u4E8E 0 \u7684\u5B89\u5168\u6574\u6570\u3002");
  }
  const decoded = [];
  let contentDecodedBytes = 0;
  for (let index = 0; index < streams.length; index += 1) {
    throwIfPdfPageAborted(options.signal);
    const bytes = await decodeStream(streams[index], pdfLib);
    throwIfPdfPageAborted(options.signal);
    if (bytes.byteLength > budget.maxDecodedStreamBytes) {
      throw pdfPageExtractionResourceLimitError(`\u7B2C ${pageNumber} \u9875\u7684\u7B2C ${index + 1} \u4E2A\u5185\u5BB9\u6D41\u89E3\u538B\u540E\u4E3A ${(bytes.byteLength / 1024 / 1024).toFixed(2)} MiB\uFF0C\u8D85\u8FC7\u5355\u6D41 ${(budget.maxDecodedStreamBytes / 1024 / 1024).toFixed(0)} MiB \u5185\u5B58\u9884\u7B97\u3002`);
    }
    contentDecodedBytes += bytes.byteLength;
    if (contentDecodedBytes > budget.maxDecodedBytes) {
      throw pdfPageExtractionResourceLimitError(`\u7B2C ${pageNumber} \u9875\u5185\u5BB9\u6D41\u89E3\u538B\u540E\u7D2F\u8BA1\u4E3A ${(contentDecodedBytes / 1024 / 1024).toFixed(2)} MiB\uFF0C\u8D85\u8FC7\u5355\u9875 ${(budget.maxDecodedBytes / 1024 / 1024).toFixed(0)} MiB \u5185\u5B58\u9884\u7B97\u3002`);
    }
    decoded.push(bytes);
  }
  const cropBox = page.getCropBox();
  const userUnitValue = page.node.getInheritableAttribute(pdfLib.PDFName.of("UserUnit"));
  const userUnitObject = userUnitValue ? page.node.context.lookup(userUnitValue) : void 0;
  const userUnit = userUnitObject instanceof pdfLib.PDFNumber ? Math.max(1e-6, userUnitObject.asNumber()) : 1;
  const resourceBudget = {
    imagePixels: 0,
    seenImages: /* @__PURE__ */ new WeakSet(),
    signal: options.signal
  };
  const resources = await extractPageResources(
    page.node.Resources(),
    pdfLib,
    /* @__PURE__ */ new Set(),
    0,
    `page:${pageNumber}`,
    resourceBudget
  );
  const annotationAppearances = await extractAnnotationAppearances(
    page.node.Annots(),
    resources,
    pdfLib,
    resourceBudget
  );
  throwIfPdfPageAborted(options.signal);
  const decodedSources = decoded.map((bytes) => byteString(bytes).replace(/\r\n?/g, "\n"));
  let contentSource = "";
  const unshiftedContentStreamRanges = [];
  decodedSources.forEach((streamSource) => {
    if (contentSource && streamSource) {
      const hasBoundaryWhitespace = /[\u0000\t\n\f\r ]$/.test(contentSource) || /^[\u0000\t\n\f\r ]/.test(streamSource);
      if (!hasBoundaryWhitespace) contentSource += "\n";
    }
    const startOffset = contentSource.length;
    contentSource += streamSource;
    unshiftedContentStreamRanges.push({
      startOffset,
      endOffset: contentSource.length
    });
  });
  const visibleSource = [contentSource, ...annotationAppearances.commands].filter(Boolean).join("\n");
  const userUnitPrefix = userUnit === 1 ? "" : `q
${userUnit} 0 0 ${userUnit} 0 0 cm
`;
  const source = userUnit === 1 ? visibleSource : `${userUnitPrefix}${visibleSource}
Q`;
  const contentStreamRanges = unshiftedContentStreamRanges.map((range) => ({
    startOffset: range.startOffset + userUnitPrefix.length,
    endOffset: range.endOffset + userUnitPrefix.length
  }));
  return {
    pageNumber,
    source,
    streamCount: streams.length,
    contentStreamRanges,
    annotationAppearanceCount: annotationAppearances.count,
    decodedBytes: contentDecodedBytes + annotationAppearances.decodedBytes,
    boxX: cropBox.x * userUnit,
    boxY: cropBox.y * userUnit,
    width: cropBox.width * userUnit,
    height: cropBox.height * userUnit,
    rotation: normalizedRotation(page.getRotation().angle),
    resources
  };
}

// arrow-sidecar/cli.mjs
var frameSize = (page) => {
  const rot = (page.rotation % 360 + 360) % 360;
  const swapped = rot === 90 || rot === 270;
  return { rot, fw: swapped ? page.height : page.width, fh: swapped ? page.width : page.height };
};
var frameToUser = (page, nx, ny) => {
  const { rot } = frameSize(page);
  const { width: W, height: H, boxX, boxY } = page;
  let ux;
  let uy;
  if (rot === 0) {
    ux = nx * W;
    uy = (1 - ny) * H;
  } else if (rot === 90) {
    ux = ny * W;
    uy = nx * H;
  } else if (rot === 180) {
    ux = (1 - nx) * W;
    uy = ny * H;
  } else {
    ux = (1 - ny) * W;
    uy = (1 - nx) * H;
  }
  return { x: boxX + ux, y: boxY + uy };
};
var userToFrame = (page, x, y) => {
  const { rot } = frameSize(page);
  const { width: W, height: H, boxX, boxY } = page;
  const ux = (x - boxX) / W;
  const uy = (y - boxY) / H;
  if (rot === 0) return { nx: ux, ny: 1 - uy };
  if (rot === 90) return { nx: uy, ny: ux };
  if (rot === 180) return { nx: 1 - ux, ny: uy };
  return { nx: 1 - uy, ny: 1 - ux };
};
var boxToUser = (page, box2d) => {
  const [ymin, xmin, ymax, xmax] = box2d;
  const a = frameToUser(page, xmin / 1e3, ymin / 1e3);
  const b = frameToUser(page, xmax / 1e3, ymax / 1e3);
  return {
    minX: Math.min(a.x, b.x),
    maxX: Math.max(a.x, b.x),
    minY: Math.min(a.y, b.y),
    maxY: Math.max(a.y, b.y)
  };
};
var pointToFrame = (page, point) => {
  const { nx, ny } = userToFrame(page, point.x, point.y);
  const clamp = (v) => Math.max(0, Math.min(1e3, Math.round(v * 1e3)));
  return [clamp(ny), clamp(nx)];
};
var boundsToFrame = (page, bounds) => {
  const a = userToFrame(page, bounds.minX, bounds.minY);
  const b = userToFrame(page, bounds.maxX, bounds.maxY);
  const clamp = (v) => Math.max(0, Math.min(1e3, Math.round(v * 1e3)));
  return [
    clamp(Math.min(a.ny, b.ny)),
    clamp(Math.min(a.nx, b.nx)),
    clamp(Math.max(a.ny, b.ny)),
    clamp(Math.max(a.nx, b.nx))
  ];
};
var readStdin = () => new Promise((resolve, reject) => {
  const chunks = [];
  process.stdin.on("data", (c) => chunks.push(c));
  process.stdin.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
  process.stdin.on("error", reject);
});
var fail = (error, code) => {
  process.stdout.write(JSON.stringify({ ok: false, error, code }));
  process.exit(0);
};
var job;
try {
  job = JSON.parse(await readStdin());
} catch (error) {
  fail(`bad job json: ${error.message}`, "BAD_JOB");
}
try {
  const bytes = new Uint8Array(await readFile(job.pdf));
  let loaded = await loadPdfBytes(bytes, { fileName: "input.pdf", fileSize: bytes.byteLength });
  const page = await extractPdfPage(
    loaded, Number(job.page), job.budget ?? void 0
  );
  loaded = null;
  const prepared = prepareCalloutPage(page, job.budget ?? void 0);
  const analysis = analyzePreparedCalloutPage({ prepared });
  const automaticCallouts = analysis.automaticDetection.callouts;
  const regions = (job.plan_regions ?? []).map((b) => boxToUser(page, b));
  const inRegion = (bounds) => regions.some((region) => bounds.minX >= region.minX && bounds.maxX <= region.maxX && bounds.minY >= region.minY && bounds.maxY <= region.maxY);
  const pointInRegion = (point) => regions.some((region) => point.x >= region.minX && point.x <= region.maxX && point.y >= region.minY && point.y <= region.maxY);
  const boxes = (job.boxes ?? []).map((b) => boxToUser(page, b));
  const gated = boxes.map((box) => regions.length === 0 || inRegion(box));
  const dropMixedBareEnds = (found) => {
    if (found.length < 2) return found;
    const heads = found.filter((r) => r.terminalKind === "arrowhead");
    return heads.length && heads.length !== found.length ? heads : found;
  };
  const anchorLabels = job.anchor_labels ?? [];
  const anchorTexts = job.anchor_texts ?? [];
  const anchorVecBacked = job.anchor_vec_backed ?? [];

  // A vector supplement can be only the final fence-bearing row of a wrapped
  // paragraph while the authored leader is attached to its first row.  Reuse
  // only a unique marked automatic owner reached through a short same-column
  // decoded-text chain; never borrow merely-nearby loose geometry.
  const decodedTextRows = prepared.scene.ops.flatMap((op, opIndex) =>
    op?.kind === "text" && op.bounds
      ? [{ opIndex, bounds: finiteBounds(op.bounds),
          frame: boundsToFrame(page, finiteBounds(op.bounds)),
          text: String(op.text ?? "") }]
      : []
  );
  const boundsOfOps = (opIndices) => {
    const rows = opIndices.map((opIndex) => prepared.scene.ops[opIndex])
      .filter((op) => op?.bounds).map((op) => finiteBounds(op.bounds));
    return rows.length
      ? rows.reduce((combined, bounds) => unionBounds(combined, bounds))
      : null;
  };
  const wordTokens = (value) => new Set(String(value ?? "").toUpperCase()
    .replace(/[^A-Z0-9]+/g, " ").trim().split(/\s+/)
    .filter((token) => token.length >= 4));
  const compactText = (value) => String(value ?? "").toUpperCase()
    .replace(/[^A-Z0-9]+/g, "");
  const axisGap = (value, low, high) =>
    Math.max(0, low - value, value - high);
  const frameAxes = (frame) => {
    const height = Math.max(1e-6, frame[2] - frame[0]);
    const width = Math.max(1e-6, frame[3] - frame[1]);
    const horizontal = width >= height;
    return horizontal ? {
      horizontal,
      shortMin: frame[0], shortMax: frame[2],
      longMin: frame[1], longMax: frame[3],
      shortSize: height,
      shortCenter: (frame[0] + frame[2]) / 2
    } : {
      horizontal,
      shortMin: frame[1], shortMax: frame[3],
      longMin: frame[0], longMax: frame[2],
      shortSize: width,
      shortCenter: (frame[1] + frame[3]) / 2
    };
  };
  const domainText = (value) => {
    const raw = String(value ?? "").toUpperCase();
    const compact = compactText(raw);
    const tokens = wordTokens(raw);
    const gateLike = compact.includes("GATE")
      && !compact.includes("AGGREGATE");
    return compact.includes("FENC") || compact.includes("CHAINLINK")
      || compact.includes("GUARDRAIL") || gateLike
      || tokens.has("GATE") || tokens.has("GATES")
      || compact === "GATE" || compact === "GATES";
  };
  const semanticallySameText = (leftValue, rightValue) => {
    const left = compactText(leftValue);
    const right = compactText(rightValue);
    if (Math.min(left.length, right.length) >= 8
        && (left.includes(right) || right.includes(left))) return true;
    const leftTokens = wordTokens(leftValue);
    const rightTokens = wordTokens(rightValue);
    if (!leftTokens.size || !rightTokens.size) return false;
    const overlap = [...leftTokens].filter((token) =>
      rightTokens.has(token)).length;
    return overlap / Math.max(leftTokens.size, rightTokens.size) >= 0.7;
  };
  const orientedFrameAxes = (frame, horizontal) => horizontal ? {
    horizontal,
    shortMin: frame[0], shortMax: frame[2],
    longMin: frame[1], longMax: frame[3],
    shortSize: Math.max(1e-6, frame[2] - frame[0]),
    shortCenter: (frame[0] + frame[2]) / 2
  } : {
    horizontal,
    shortMin: frame[1], shortMax: frame[3],
    longMin: frame[0], longMax: frame[2],
    shortSize: Math.max(1e-6, frame[3] - frame[1]),
    shortCenter: (frame[1] + frame[3]) / 2
  };
  const buildDecodedLines = (horizontal) => {
    const glyphs = decodedTextRows.map((row) => ({
      ...row,
      axes: orientedFrameAxes(row.frame, horizontal)
    })).sort((left, right) => left.axes.shortCenter - right.axes.shortCenter
      || left.axes.longMin - right.axes.longMin || left.opIndex - right.opIndex);
    const bands = [];
    for (const glyph of glyphs) {
      const held = bands[bands.length - 1];
      const tolerance = held
        ? Math.max(held.shortSize, glyph.axes.shortSize) * 0.45 : 0;
      if (!held || glyph.axes.shortCenter - held.shortCenter > tolerance) {
        bands.push({ glyphs: [glyph], shortCenter: glyph.axes.shortCenter,
          shortSize: glyph.axes.shortSize });
      } else {
        held.glyphs.push(glyph);
        held.shortCenter = held.glyphs.reduce((sum, row) =>
          sum + row.axes.shortCenter, 0) / held.glyphs.length;
        held.shortSize = Math.max(held.shortSize, glyph.axes.shortSize);
      }
    }
    const lines = [];
    for (const band of bands) {
      const ordered = band.glyphs.sort((left, right) =>
        left.axes.longMin - right.axes.longMin || left.opIndex - right.opIndex);
      let segment = [];
      const publish = () => {
        if (!segment.length) return;
        const frame = segment.map((row) => row.frame)
          .reduce((combined, row) => [
            Math.min(combined[0], row[0]), Math.min(combined[1], row[1]),
            Math.max(combined[2], row[2]), Math.max(combined[3], row[3])
          ]);
        lines.push({
          opIndex: Math.min(...segment.map((row) => row.opIndex)),
          opIndices: segment.map((row) => row.opIndex),
          frame,
          text: segment.map((row) => row.text).join(""),
          axes: orientedFrameAxes(frame, horizontal)
        });
        segment = [];
      };
      for (const glyph of ordered) {
        const previous = segment[segment.length - 1];
        if (previous) {
          const longGap = glyph.axes.longMin - previous.axes.longMax;
          const splitGap = Math.max(4,
            previous.axes.shortSize * 2, glyph.axes.shortSize * 2);
          if (longGap > splitGap) publish();
        }
        segment.push(glyph);
      }
      publish();
    }
    return lines;
  };
  const decodedOwnerLines = {};
  const ownerLinesFor = (horizontal) => {
    const key = horizontal ? "horizontal" : "vertical";
    if (!decodedOwnerLines[key]) {
      decodedOwnerLines[key] = buildDecodedLines(horizontal);
    }
    return decodedOwnerLines[key];
  };
  const spatialOwnershipDebug = [];

  // A geometry-only recovery must not borrow a real leader whose root belongs
  // to a closer, non-fence text paragraph omitted from the supplied union.  A
  // common failure is two adjacent notes in the same column: both marked
  // packets pass the generous root radius, so the fence anchor receives the
  // aluminium/concrete note's leader as a second branch.  Compare the visual
  // short axis (Y for horizontal text, X for vertical text), then reconstruct
  // the seed row's compact paragraph.  Repeated fence/gate wording stays
  // eligible because the UI intentionally merges identical callouts.
  const foreignDecodedTextOwnsRoot = (root, anchorIndex) => {
    if (!domainText(anchorTexts[anchorIndex])) return false;
    const targetFrame = boundsToFrame(page, boxes[anchorIndex]);
    const target = frameAxes(targetFrame);
    const [rootY, rootX] = pointToFrame(page, root);
    const shortValue = target.horizontal ? rootY : rootX;
    const longValue = target.horizontal ? rootX : rootY;
    const targetShortGap = axisGap(
      shortValue, target.shortMin, target.shortMax
    );
    const ownershipMargin = Math.max(
      1.2, Math.min(4, target.shortSize * 0.25)
    );
    if (targetShortGap < Math.max(6, ownershipMargin)) {
      if (job.debug_spatial_candidates) spatialOwnershipDebug.push({
        anchor_index: anchorIndex, root: [rootY, rootX],
        target_short_gap: targetShortGap, reason: "target-short-axis"
      });
      return false;
    }

    const ownerLines = ownerLinesFor(target.horizontal);
    const nearby = ownerLines.filter((row) => {
      const longGap = axisGap(
        longValue, row.axes.longMin, row.axes.longMax
      );
      return longGap <= Math.max(18, row.axes.shortSize * 8,
        target.shortSize * 2.4);
    }).map((row) => ({
      ...row,
      shortGap: axisGap(shortValue, row.axes.shortMin, row.axes.shortMax),
      longGap: axisGap(longValue, row.axes.longMin, row.axes.longMax)
    })).sort((left, right) => left.shortGap - right.shortGap
      || left.longGap - right.longGap || left.opIndex - right.opIndex);
    const seed = nearby[0];
    if (!seed || seed.shortGap + ownershipMargin >= targetShortGap) {
      if (job.debug_spatial_candidates) spatialOwnershipDebug.push({
        anchor_index: anchorIndex, root: [rootY, rootX],
        target_short_gap: targetShortGap,
        reason: seed ? "no-closer-decoded-row" : "no-decoded-row",
        ...seed ? { seed: { text: seed.text, frame: seed.frame,
          short_gap: seed.shortGap, long_gap: seed.longGap } } : {}
        , nearest_any: ownerLines.map((row) => ({
          text: row.text, frame: row.frame,
          dy: axisGap(rootY, row.frame[0], row.frame[2]),
          dx: axisGap(rootX, row.frame[1], row.frame[3])
        })).sort((left, right) => Math.hypot(left.dy, left.dx)
          - Math.hypot(right.dy, right.dx)).slice(0, 6)
      });
      return false;
    }

    const columnTolerance = Math.max(3,
      seed.axes.shortSize * 2.2);
    const column = ownerLines.filter((row) =>
      Math.abs(row.axes.longMin - seed.axes.longMin) <= columnTolerance
      && row.axes.shortSize / seed.axes.shortSize >= 0.45
      && row.axes.shortSize / seed.axes.shortSize <= 2.2
    ).sort((left, right) => left.axes.shortCenter - right.axes.shortCenter
      || left.opIndex - right.opIndex);
    let seedAt = column.findIndex((row) => row.opIndex === seed.opIndex);
    if (seedAt < 0) return false;
    let firstRow = seedAt;
    let lastRow = seedAt;
    while (firstRow > 0 && lastRow - firstRow + 1 < 5) {
      const previous = column[firstRow - 1];
      const current = column[firstRow];
      const maxStep = Math.max(previous.axes.shortSize,
        current.axes.shortSize) * 1.85;
      if (current.axes.shortCenter - previous.axes.shortCenter > maxStep) break;
      firstRow -= 1;
    }
    while (lastRow + 1 < column.length && lastRow - firstRow + 1 < 5) {
      const current = column[lastRow];
      const next = column[lastRow + 1];
      const maxStep = Math.max(current.axes.shortSize,
        next.axes.shortSize) * 1.85;
      if (next.axes.shortCenter - current.axes.shortCenter > maxStep) break;
      lastRow += 1;
    }
    const ownerText = column.slice(firstRow, lastRow + 1)
      .map((row) => row.text).join(" ");
    const blocked = !domainText(ownerText)
      || !semanticallySameText(ownerText, anchorTexts[anchorIndex]);
    if (job.debug_spatial_candidates) spatialOwnershipDebug.push({
      anchor_index: anchorIndex,
      root: [rootY, rootX],
      target_short_gap: targetShortGap,
      seed: { text: seed.text, frame: seed.frame,
        short_gap: seed.shortGap, long_gap: seed.longGap },
      owner_text: ownerText,
      blocked
    });
    return blocked;
  };
  const resolveWrappedParagraph = (resolution, index) => {
    if (resolution.source === "automatic") return resolution;
    if (String(anchorLabels[index] ?? "").trim().toLowerCase()
        !== "vector supplement") return resolution;
    const anchorText = String(anchorTexts[index] ?? "");
    if (!/(?:FENC\w*|GUARDRAIL\w*|GATE\w*)/i.test(anchorText)) {
      return resolution;
    }
    if (/\b(?:DETAIL|SECTION|ELEVATION|PLAN)\b/i.test(anchorText)) {
      return resolution;
    }
    const seedOps = (resolution.callout.textOps ?? [])
      .filter((opIndex) => prepared.scene.ops[opIndex]?.kind === "text");
    const seedBounds = boundsOfOps(seedOps);
    if (!seedBounds) return resolution;
    const suppliedTokens = wordTokens(anchorText);
    const decodedTokens = wordTokens(seedOps.map((opIndex) =>
      prepared.scene.ops[opIndex]?.text ?? "").join(" "));
    const suppliedCompact = compactText(anchorText);
    const decodedCompact = compactText(seedOps.map((opIndex) =>
      prepared.scene.ops[opIndex]?.text ?? "").join(""));
    const tokenMatch = [...suppliedTokens]
      .some((token) => decodedTokens.has(token));
    const compactMatch = decodedCompact.length >= 4
      && (suppliedCompact.includes(decodedCompact)
        || decodedCompact.includes(suppliedCompact));
    if (!tokenMatch && !compactMatch) {
      return resolution;
    }
    // Compare paragraph rows in the public display frame.  PDF user-space X/Y
    // swap on 90/270-degree sheets; frame coordinates keep "same left edge"
    // and "next visual row" invariant under page rotation.
    const seedFrame = boundsToFrame(page, seedBounds);
    const seedHeight = Math.max(1, seedFrame[2] - seedFrame[0]);
    const seedCenterY = (seedFrame[0] + seedFrame[2]) / 2;
    const candidates = [];
    for (const callout of automaticCallouts) {
      if (!(callout.leaders?.length ?? 0)) continue;
      const carrierOps = (callout.textOps ?? [])
        .filter((opIndex) => prepared.scene.ops[opIndex]?.kind === "text");
      const carrierBounds = boundsOfOps(carrierOps);
      if (!carrierBounds) continue;
      const carrierFrame = boundsToFrame(page, carrierBounds);
      const carrierHeight = Math.max(1, carrierFrame[2] - carrierFrame[0]);
      const heightRatio = carrierHeight / seedHeight;
      if (heightRatio < 0.55 || heightRatio > 1.8) continue;
      const rowHeight = Math.max(seedHeight, carrierHeight);
      const xTolerance = Math.max(rowHeight * 0.9, 1.2);
      if (Math.abs(carrierFrame[1] - seedFrame[1]) > xTolerance) continue;
      const carrierCenterY = (carrierFrame[0] + carrierFrame[2]) / 2;
      if (Math.abs(carrierCenterY - seedCenterY) > rowHeight * 4.4) continue;
      const low = Math.min(carrierCenterY, seedCenterY) - rowHeight * 0.35;
      const high = Math.max(carrierCenterY, seedCenterY) + rowHeight * 0.35;
      const centers = decodedTextRows.filter((row) => {
        const height = Math.max(1, row.frame[2] - row.frame[0]);
        const ratio = height / rowHeight;
        const center = (row.frame[0] + row.frame[2]) / 2;
        return ratio >= 0.5 && ratio <= 1.55
          && center >= low && center <= high
          && Math.abs(row.frame[1] - seedFrame[1]) <= xTolerance;
      }).map((row) => (row.frame[0] + row.frame[2]) / 2)
        .sort((left, right) => left - right);
      const rows = [];
      for (const center of centers) {
        if (!rows.length || center - rows[rows.length - 1] > rowHeight * 0.35) {
          rows.push(center);
        }
      }
      if (rows.length < 2 || rows.length > 5) continue;
      if (rows.some((center, rowIndex) => rowIndex > 0
          && center - rows[rowIndex - 1] > rowHeight * 2.0)) continue;
      candidates.push(callout);
    }
    const unique = new Map(candidates.map((callout) => [callout.id, callout]));
    if (unique.size !== 1) return resolution;
    const callout = [...unique.values()][0];
    return {
      ...resolution,
      source: "automatic",
      callout,
      hasLeader: callout.leaders.length > 0,
      intersectingOps: seedOps,
      targetRegions: buildCalloutTargetRegions(
        prepared.scene, [callout], prepared.pageBounds
      )
    };
  };
  const first = resolveCalloutBoxes(prepared, boxes, { automaticCallouts })
    .map(resolveWrappedParagraph);
  const claimants = /* @__PURE__ */ new Map();
  first.forEach((resolution, index) => {
    if (resolution.source !== "automatic") return;
    const carrier = resolution.callout.textOps.length ? resolution.callout.textOps : resolution.callout.textFrameOps;
    const box = boxes[index];
    const covered = carrier.filter((opIndex) => {
      const b = prepared.scene.ops[opIndex]?.bounds;
      return b && b.minX <= box.maxX && b.maxX >= box.minX && b.minY <= box.maxY && b.maxY >= box.minY;
    }).length;
    const strength = carrier.length ? covered / carrier.length : 0;
    const held = claimants.get(resolution.callout.id);
    if (!held || strength > held.strength) {
      claimants.set(resolution.callout.id, { index, strength });
    }
  });
  let displaced = 0;
  const resolutions = first.map((resolution, index) => {
    if (resolution.source !== "automatic") return resolution;
    const holder = claimants.get(resolution.callout.id);
    if (holder && holder.index === index) return resolution;
    displaced += 1;
    return resolveCalloutBox(prepared, boxes[index], {});
  });
  const spatialCandidateDebug = job.debug_spatial_candidates ? (() => {
    const pageDiagonal = Math.max(1, boundsDiagonal(prepared.pageBounds));
    return leaderCandidates(
      prepared.scene,
      pageDiagonal,
      prepared.segmentation
    ).filter((leader) => leader.packetKind !== "leader-only" || leader.markerOps.length)
      .map((leader) => {
        const markerBounds = leader.markerOps.length ? leader.markerOps
          .map((opIndex) => finiteBounds(prepared.scene.ops[opIndex].bounds))
          .reduce((combined, bounds) => unionBounds(combined, bounds)) : null;
        const markerEnd = markerBounds && pointToBounds(leader.endpoints[1], markerBounds)
          < pointToBounds(leader.endpoints[0], markerBounds) ? 1 : 0;
        const rootEnd = markerEnd === 0 ? 1 : 0;
        return {
          op_index: leader.opIndex,
          packet_kind: leader.packetKind,
          marker_kind: leader.markerKind,
          path_ops: leader.pathOps,
          arrow_ops: leader.arrowheadOps,
          marker_ops: leader.markerOps,
          marker_end: markerEnd,
          endpoints: leader.endpoints.map((point) => pointToFrame(page, point)),
          route: pointDistance(leader.endpoints[0], leader.endpoints[1]),
          distances: boxes.map((box) => pointToBounds(leader.endpoints[rootEnd], box))
        };
      });
  })() : void 0;
  const roiCallouts = resolutions.map((resolution, index) => {
    if (resolution.source !== "roi") return null;
    const inspection = inspectCalloutTextRoi(
      prepared.scene,
      prepared.segmentation,
      boxes[index],
      {}
    );
    const found = inspection?.callouts ?? [];
    return found.length > 1 ? found : null;
  });

  // One automatic text cluster may contain two adjacent decoded rows even
  // though the caller supplied a box for only one of them.  Keep a leader
  // only when its authored root belongs to a decoded text op covered by the
  // supplied box.  This is deliberately component-level: branches that share
  // a root keep the same owner, while a neighbouring note such as NEW
  // CONCRETE WALK cannot donate its independent leader to NEW CHAIN LINK
  // FENCE merely because both rows were decoded into one cluster.
  const scopeCalloutLeadersToBox = (callout, box) => {
    const carrierOps = (callout.textOps ?? [])
      .filter((opIndex) => prepared.scene.ops[opIndex]?.kind === "text");
    // A single leader attached to a wrapped paragraph belongs to the whole
    // paragraph even when the supplied fence box covers only its final line.
    // Row-level splitting is meaningful only when the merged carrier itself
    // owns multiple independent leader components.
    if (carrierOps.length < 2 || (callout.leaders?.length ?? 0) < 2) return callout;
    const ownedOps = carrierOps.filter((opIndex) =>
      overlaps(prepared.scene.ops[opIndex].bounds, box));
    if (!ownedOps.length || ownedOps.length === carrierOps.length) return callout;
    const foreignOps = carrierOps.filter((opIndex) => !ownedOps.includes(opIndex));
    const ownershipMargin = Math.max(
      1e-7, boundsDiagonal(prepared.pageBounds) * 5e-4
    );
    const distanceToOps = (point, opIndices) => Math.min(...opIndices.map(
      (opIndex) => pointToBounds(point, prepared.scene.ops[opIndex].bounds)
    ));
    const leaders = callout.leaders.filter((leader) => {
      const root = leader.root;
      if (!root) return true;
      const ownedDistance = distanceToOps(root, ownedOps);
      const foreignDistance = distanceToOps(root, foreignOps);
      return foreignDistance + ownershipMargin >= ownedDistance;
    });
    return leaders.length === callout.leaders.length
      ? callout : { ...callout, leaders };
  };
  const calloutsAt = resolutions.map((resolution, index) =>
    (roiCallouts[index] ?? [resolution.callout])
      .map((callout) => scopeCalloutLeadersToBox(callout, boxes[index])));

  // Spatial marked-leader recovery.  The primary detector deliberately uses
  // PDF paint order to keep table rules and title underlines out.  Some CAD
  // writers batch all labels and all arrows separately, though, so a real
  // leader can be geometrically attached to its text while being thousands of
  // paint operations away.  Recover only candidates that already carry an
  // explicit arrow/open-marker packet.  It is allowed only for an unresolved
  // supplied callout, and every leader already authored to *any* automatic
  // callout on the page is globally owned.  Thus a nearby non-fence note can
  // block reassignment even though that note was not sent as an output anchor.
  const spatialFallbackByIndex = (() => {
    const scene = prepared.scene;
    const pageDiagonal = Math.max(1, boundsDiagonal(prepared.pageBounds));
    const regularLeadersAt = calloutsAt.map((callouts) =>
      callouts.flatMap((callout) => callout.leaders));
    const leaderOps = (leader) => [
      ...(leader.pathOps ?? []),
      ...(leader.arrowheadOps ?? []),
      ...(leader.markerOps ?? []),
      ...(leader.targets ?? []).flatMap((target) => target.markerOps ?? [])
    ];
    const claimedOps = new Set(automaticCallouts.flatMap((callout) =>
      (callout.leaders ?? []).flatMap(leaderOps)));

    // The supplied VLM/vector-union boxes are the stable ownership frame.
    // resolveCalloutBox may deliberately fall back to a nearby decoded text
    // cluster; using that inferred carrier here can jump across adjacent legend
    // rows, exactly the failure this fallback must avoid.
    const eligible = boxes.flatMap((bounds, index) => {
      const label = String(anchorLabels[index] ?? "").toLowerCase();
      // Vector supplements include table rows and detail titles.  They need
      // the stricter endpoint-graph pass in Python; this sidecar-only pass is
      // reserved for model-confirmed callouts.
      if (label !== "callout") return [];
      // Never append a second, merely-nearby component to a callout that the
      // authored detector has already resolved.  Genuine multi-branch leaders
      // are emitted together by the primary detector; recovery is for empty
      // anchors only.
      if (regularLeadersAt[index].length !== 0) return [];
      // Proximity alone cannot validate an unbacked VLM-only anchor when the
      // resolver found neither decoded text nor automatic/ROI ownership.  It
      // would otherwise borrow a neighbouring annotation's real arrow.  Keep
      // vec-backed outline callouts eligible; they are known positive text
      // even when this PDF decoder cannot expose their glyphs.
      const resolution = resolutions[index];
      if (resolution?.source === "text-only"
          && !(resolution.callout.textOps?.length ?? 0)
          && anchorVecBacked[index] !== true) return [];
      const width = boundsWidth(bounds);
      const height = boundsHeight(bounds);
      const scale = Math.max(1, Math.min(width, height));
      const threshold = Math.min(
        pageDiagonal * 0.018,
        Math.max(scale * 4, pageDiagonal * 25e-4)
      );
      return [{ index, bounds, label, scale, threshold }];
    });
    if (!eligible.length) return new Map();

    const recovered = new Map();
    const branchCount = regularLeadersAt.map((leaders) => leaders.length);
    const candidates = leaderCandidates(
      scene, pageDiagonal, prepared.segmentation
    ).filter((leader) => (leader.markerOps?.length ?? 0) > 0
      && leader.packetKind !== "leader-only"
      && ![...(leader.pathOps ?? []), ...(leader.markerOps ?? [])]
        .some((opIndex) => claimedOps.has(opIndex)));

    for (const leader of candidates) {
      const markerBounds = leader.markerOps
        .map((opIndex) => finiteBounds(scene.ops[opIndex].bounds))
        .reduce((combined, bounds) => unionBounds(combined, bounds));
      const firstMarkerDistance = pointToBounds(leader.endpoints[0], markerBounds);
      const secondMarkerDistance = pointToBounds(leader.endpoints[1], markerBounds);
      const markerEndIndex = secondMarkerDistance < firstMarkerDistance ? 1 : 0;
      const root = leader.endpoints[markerEndIndex === 0 ? 1 : 0];
      const markerEnd = leader.endpoints[markerEndIndex];
      const routeLength = pointDistance(root, markerEnd);
      const ranked = eligible.map((anchor) => {
        const rootDistance = pointToBounds(root, anchor.bounds);
        return { ...anchor, rootDistance,
          score: rootDistance / anchor.threshold };
      }).filter((anchor) => anchor.score <= 1
        && routeLength >= pageDiagonal * 1e-3)
        .sort((left, right) => left.score - right.score
          || left.rootDistance - right.rootDistance || left.index - right.index);
      if (!ranked.length) continue;
      const best = ranked[0];
      const second = ranked[1];
      // A marker exactly between two labels is not safe to assign.  Callout
      // boxes get a slightly smaller margin because multiline boxes often end
      // short of their authored leader root; vector supplements stay strict.
      const uniqueGap = pageDiagonal * 35e-5;
      // An ineligible title/legend/note can still be the geometric owner of a
      // marked leader.  It must block reassignment to a farther eligible box;
      // otherwise a legend's sample arrow can jump to the nearest callout.
      const globalOwner = boxes.map((bounds, index) => ({
        index,
        distance: pointToBounds(root, bounds)
      })).sort((left, right) => left.distance - right.distance
        || left.index - right.index)[0];
      if (globalOwner && globalOwner.index !== best.index
          && globalOwner.distance < best.rootDistance + uniqueGap) continue;
      if (foreignDecodedTextOwnsRoot(root, best.index)) continue;
      if (second && second.rootDistance - best.rootDistance < uniqueGap
          && second.score < best.score * 1.15) continue;
      if (branchCount[best.index] >= 4) continue;
      const owned = recovered.get(best.index) ?? [];
      owned.push({ leader, root, markerEnd, markerBounds });
      recovered.set(best.index, owned);
      branchCount[best.index] += 1;
      leaderOps(leader).forEach((opIndex) => claimedOps.add(opIndex));
    }
    return recovered;
  })();

  const opPolylines = (opIndices) => {
    const out = [];
    for (const opIndex of opIndices) {
      const op = prepared.scene.ops[opIndex];
      if (op?.kind !== "path") continue;
      let current = [];
      for (const segment of op.segments) {
        if (segment.kind === "move") {
          if (current.length > 1) out.push(current);
          current = [pointToFrame(page, segment)];
        } else if (segment.kind === "line") {
          current.push(pointToFrame(page, segment));
        } else if (segment.kind === "curve") {
          current.push(pointToFrame(page, { x: segment.x1, y: segment.y1 }));
          current.push(pointToFrame(page, { x: segment.x2, y: segment.y2 }));
          current.push(pointToFrame(page, segment));
        } else if (segment.kind === "close" && current.length) {
          current.push(current[0]);
        }
      }
      if (current.length > 1) out.push(current);
    }
    return out;
  };
  const results = resolutions.map((resolution, index) => {
    if (!gated[index]) {
      return {
        index,
        source: "outside-plan",
        has_leader: false,
        text: resolution.callout.text ?? "",
        leader_count: 0,
        leader_strokes: [],
        arrow_strokes: [],
        targets: []
      };
    }
    const callouts = calloutsAt[index];
    const leaders = callouts.flatMap((callout) => callout.leaders);
    const spatial = spatialFallbackByIndex.get(index) ?? [];
    const leaderStrokes = [
      ...leaders.flatMap((leader) => opPolylines(leader.pathOps)),
      ...spatial.flatMap(({ leader }) => opPolylines(leader.pathOps))
    ];
    const arrowStrokes = [
      ...leaders.flatMap((leader) => opPolylines(leader.arrowheadOps)),
      ...spatial.flatMap(({ leader }) => opPolylines(leader.arrowheadOps))
    ];
    const targetRegions = dropMixedBareEnds(
      buildCalloutTargetRegions(
        prepared.scene,
        callouts,
        prepared.pageBounds
      ).filter((region) => regions.length === 0 || pointInRegion(region.center))
    );
    const spatialTargets = spatial.map(({ leader, root, markerEnd, markerBounds }) => {
      const target = resolveLeaderTarget(
        prepared.scene, leader, root, markerEnd,
        Math.max(1, boundsDiagonal(prepared.pageBounds))
      );
      return {
        terminalKind: "arrowhead",
        center: target,
        bounds: markerBounds
      };
    }).filter((region) => regions.length === 0 || pointInRegion(region.center));
    const debug = job.debug_anchors ? (() => {
      const insp = inspectCalloutTextRoi(prepared.scene, prepared.segmentation, boxes[index], {});
      const roiCalloutCount = insp?.callouts?.length ?? 0;
      const carrier = resolution.callout.textOps.length ? resolution.callout.textOps : resolution.callout.textFrameOps;
      const cb = carrier.length ? carrier.reduce((acc, i) => {
        const b = prepared.scene.ops[i]?.bounds;
        if (!b) return acc;
        return acc ? {
          minX: Math.min(acc.minX, b.minX),
          minY: Math.min(acc.minY, b.minY),
          maxX: Math.max(acc.maxX, b.maxX),
          maxY: Math.max(acc.maxY, b.maxY)
        } : { ...b };
      }, null) : null;
      const gapTo = (i) => carrier.length ? Math.min(...carrier.map((c) => Math.abs(c - i))) : null;
      const distTo = (i) => {
        const b = prepared.scene.ops[i]?.bounds;
        if (!b || !cb) return null;
        const dx = Math.max(0, Math.max(cb.minX - b.maxX, b.minX - cb.maxX));
        const dy = Math.max(0, Math.max(cb.minY - b.maxY, b.minY - cb.maxY));
        return Math.round(Math.hypot(dx, dy) * 10) / 10;
      };
      const bnds = (ops) => {
        let r = null;
        for (const i of ops) {
          const b = prepared.scene.ops[i]?.bounds;
          if (!b) continue;
          r = r ? {
            minX: Math.min(r.minX, b.minX),
            minY: Math.min(r.minY, b.minY),
            maxX: Math.max(r.maxX, b.maxX),
            maxY: Math.max(r.maxY, b.maxY)
          } : { ...b };
        }
        return r;
      };
      const gap = (box, i) => {
        const b = prepared.scene.ops[i]?.bounds;
        if (!b || !box) return null;
        const dx = Math.max(0, Math.max(box.minX - b.maxX, b.minX - box.maxX));
        const dy = Math.max(0, Math.max(box.minY - b.maxY, b.minY - box.maxY));
        return Math.round(Math.hypot(dx, dy) * 10) / 10;
      };
      const inputBox = boxes[index];
      const carrierBox = bnds(carrier);
      const textBox = bnds(resolution.callout.textOps);
      return {
        roi_callout_count: roiCalloutCount,
        roi_callout_leaders: (insp?.callouts ?? []).map((c) => c.leaders.length),
        carrier_ops: carrier.slice(0, 6),
        carrier_is_text: resolution.callout.textOps.length > 0,
        // 三个参照系：调用方给的输入框 / 算法认定的载体 / 其中的纯文字部分
        input_box: inputBox ? boundsToFrame(page, inputBox) : null,
        carrier_box: carrierBox ? boundsToFrame(page, carrierBox) : null,
        text_box: textBox ? boundsToFrame(page, textBox) : null,
        evidence: resolution.callout.evidence,
        marker_ops: [...new Set(leaders.flatMap((l) => l.targets.flatMap((t) => t.markerOps || [])))].slice(0, 8),
        arrowhead_ops: [...new Set(leaders.flatMap((l) => l.arrowheadOps))].slice(0, 8),
        leaders: leaders.map((l) => ({
          path_ops: l.pathOps.slice(0, 6),
          arrow_ops: l.arrowheadOps.slice(0, 4),
          seq_gap: l.pathOps.length ? Math.min(...l.pathOps.map(gapTo)) : null,
          d_input: l.pathOps.length ? Math.min(...l.pathOps.map((i) => gap(inputBox, i)).filter((v) => v !== null)) : null,
          d_carrier: l.pathOps.length ? Math.min(...l.pathOps.map((i) => gap(carrierBox, i)).filter((v) => v !== null)) : null,
          d_text: textBox && l.pathOps.length ? Math.min(...l.pathOps.map((i) => gap(textBox, i)).filter((v) => v !== null)) : null,
          terminals: l.targets.map((t) => t.terminalKind)
        }))
      };
    })() : void 0;
    return {
      index,
      source: resolution.source,
      carrier_is_text: (resolution.callout.textOps?.length ?? 0) > 0,
      ...debug ? { debug } : {},
      has_leader: targetRegions.length + spatialTargets.length > 0,
      text: resolution.callout.text ?? "",
      leader_count: leaders.length + spatial.length,
      callout_count: callouts.length,
      // Painted leader / arrowhead strokes in page frame, for colouring.
      leader_strokes: leaderStrokes,
      arrow_strokes: arrowStrokes,
      // Every terminal is described the same way — a callout with four leaders
      // gets four fully-populated entries, not one plus three stubs.
      targets: [...targetRegions, ...spatialTargets].map((region, ordinal) => ({
        ordinal: region.ordinal ?? ordinal,
        terminal_kind: region.terminalKind,
        tip: pointToFrame(page, region.center),
        box_2d: boundsToFrame(page, region.bounds)
      }))
    };
  });
  const automatic = job.emit_automatic ? automaticCallouts.map((callout) => {
    const ops = callout.textOps.length ? callout.textOps : callout.textFrameOps;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const opIndex of ops) {
      const b = prepared.scene.ops[opIndex]?.bounds;
      if (!b) continue;
      minX = Math.min(minX, b.minX);
      minY = Math.min(minY, b.minY);
      maxX = Math.max(maxX, b.maxX);
      maxY = Math.max(maxY, b.maxY);
    }
    return {
      text: (callout.text ?? "").slice(0, 40),
      leaders: callout.leaders.length,
      box_2d: Number.isFinite(minX) ? boundsToFrame(page, { minX, minY, maxX, maxY }) : null
    };
  }) : void 0;
  process.stdout.write(JSON.stringify({
    ok: true,
    page: {
      width: page.width,
      height: page.height,
      rotation: page.rotation,
      ops: prepared.scene.ops.length,
      automatic_callouts: automaticCallouts.length,
      displaced_claims: displaced
    },
    ...automatic ? { automatic } : {},
    ...spatialCandidateDebug ? { spatial_candidates: spatialCandidateDebug } : {},
    ...job.debug_spatial_candidates ? {
      spatial_ownership: spatialOwnershipDebug
    } : {},
    ...job.debug_spatial_candidates ? {
      spatial_recovered: [...spatialFallbackByIndex].map(([index, rows]) => ({
        index,
        ops: rows.map(({ leader }) => leader.opIndex)
      }))
    } : {},
    results
  }));
} catch (error) {
  const code = error?.code === "CALLOUT_PAGE_RESOURCE_LIMIT" ? "PAGE_TOO_LARGE" : error?.name === "PdfPageExtractionError" ? "PDF_ERROR" : "ALGO_ERROR";
  fail(`${error?.name ?? "Error"}: ${error?.message ?? error}`, code);
}
