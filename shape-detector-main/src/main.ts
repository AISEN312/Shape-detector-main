import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * Algorithm approach:
   * 1. Edge detection (Sobel operator) for shape boundaries
   * 2. Binary thresholding with Otsu's method
   * 3. Connected component labeling
   * 4. Corner detection (Harris corners) for vertex identification
   * 5. Contour analysis and convex hull
   * 6. Geometric & mathematical shape classification
   * 7. Pattern recognition for distinguishing similar shapes
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    // Step 1: Edge detection (used as a secondary signal)
    const edges = this.detectEdges(imageData);
    
    // Step 2: Convert ImageData to binary image with Otsu's thresholding
    const binaryImage = this.toBinaryImage(imageData);
    
    // Step 3: Find connected components (shapes)
    const components = this.findConnectedComponents(binaryImage, imageData.width);
    
    // Step 4: Filter and classify shapes
    const shapes: DetectedShape[] = [];
    
    for (const component of components) {
      if (component.pixels.length < 20) continue; // Skip very small components (noise)
      
      const shape = this.classifyShape(component, edges, imageData.width);
      if (shape && shape.confidence > 0.55) { // Require confidence > 0.55
        shapes.push(shape);
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

  /**
   * Sobel edge detection operator
   */
  private detectEdges(imageData: ImageData): Uint8Array {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const edges = new Uint8Array(width * height);

    // Sobel kernels
    const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let gx = 0, gy = 0;

        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx = ((y + ky) * width + (x + kx)) * 4;
            const lum = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
            
            gx += lum * sobelX[ky + 1][kx + 1];
            gy += lum * sobelY[ky + 1][kx + 1];
          }
        }

        const magnitude = Math.sqrt(gx * gx + gy * gy);
        edges[y * width + x] = Math.min(255, magnitude > 128 ? 255 : 0);
      }
    }

    return edges;
  }

  /**
   * Convert ImageData to binary image (black/white)
   */
  private toBinaryImage(imageData: ImageData): Uint8Array {
    const data = imageData.data;
    const length = imageData.width * imageData.height;
    const binary = new Uint8Array(length);

    // Calculate histogram for auto-threshold
    const histogram = new Uint32Array(256);
    for (let i = 0; i < length; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const luminance = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      histogram[luminance]++;
    }

    // Otsu's method for automatic threshold
    let threshold = 128;
    let maxVariance = 0;
    let sumB = 0;
    let wB = 0;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
      sum += i * histogram[i];
    }
    
    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      const wF = length - wB;
      if (wF === 0) break;
      
      sumB += t * histogram[t];
      const meanB = sumB / wB;
      const meanF = (sum - sumB) / wF;
      const variance = wB * wF * Math.pow(meanB - meanF, 2);
      
      if (variance > maxVariance) {
        maxVariance = variance;
        threshold = t;
      }
    }

    for (let i = 0; i < length; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
      binary[i] = luminance > threshold ? 255 : 0;
    }

    return binary;
  }

  /**
   * Find connected components (potential shapes)
   */
  private findConnectedComponents(binary: Uint8Array, width: number) {
    const visited = new Uint8Array(binary.length);
    const components: Array<{
      pixels: Array<{x: number; y: number}>;
      isBlack: boolean;
    }> = [];

    for (let i = 0; i < binary.length; i++) {
      if (!visited[i] && binary[i] === 0) { // Black pixels are shapes
        const component = this.floodFill(binary, visited, i, width);
        if (component.pixels.length > 20) { // Increased minimum size to filter noise
          components.push(component);
        }
      }
    }

    return components;
  }

  /**
   * Flood fill algorithm to find connected components
   */
  private floodFill(binary: Uint8Array, visited: Uint8Array, start: number, width: number) {
    const pixels: Array<{x: number; y: number}> = [];
    const queue: number[] = [start];
    visited[start] = 1;

    while (queue.length > 0) {
      const idx = queue.shift()!;
      const y = Math.floor(idx / width);
      const x = idx % width;

      pixels.push({x, y});

      // Check 8-connected neighbors
      const neighbors = [
        idx - width - 1, idx - width, idx - width + 1,
        idx - 1, idx + 1,
        idx + width - 1, idx + width, idx + width + 1
      ];

      for (const nIdx of neighbors) {
        if (nIdx >= 0 && nIdx < binary.length && !visited[nIdx] && binary[nIdx] === 0) {
          visited[nIdx] = 1;
          queue.push(nIdx);
        }
      }
    }

    return {pixels, isBlack: true};
  }

  /**
   * Classify a shape based on geometric properties
   */
  private classifyShape(component: any, edges: Uint8Array, imageWidth: number): DetectedShape | null {
    const pixels = component.pixels;
    
    // Calculate bounding box
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    
    for (const p of pixels) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }

    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    const area = pixels.length;
    const boundingArea = width * height;

    // Fast rejection for tiny/thin noise components.
    if (area < 120 || width < 10 || height < 10) {
      return null;
    }

    // Calculate center
    let centerX = 0, centerY = 0;
    for (const p of pixels) {
      centerX += p.x;
      centerY += p.y;
    }
    centerX /= pixels.length;
    centerY /= pixels.length;

    // Extract boundary metrics and reject line/text-like components.
    const { boundaryPixels, perimeter } = this.extractBoundaryAndPerimeter(pixels);
    if (boundaryPixels.length < 20 || perimeter <= 0) {
      return null;
    }

    const extent = area / Math.max(1, boundingArea);
    const thinness = (perimeter * perimeter) / Math.max(1, 4 * Math.PI * area);
    if (extent < 0.16 || thinness > 20) {
      return null;
    }

    // Radial-angle signature: circles have stable radius across angles.
    const radial = this.analyzeRadialSignature(boundaryPixels, centerX, centerY);
    if (radial.coverage < 0.45) {
      return null;
    }

    // Corner analysis from hull geometry.
    const hull = this.getConvexHull(boundaryPixels);
    const corners = this.detectCorners(hull, centerX, centerY);
    const polygonSides = this.estimatePolygonSides(hull, Math.max(2, perimeter * 0.02));

    // Estimate edge support around this component.
    let edgeHits = 0;
    for (const p of boundaryPixels) {
      if (edges[p.y * imageWidth + p.x] > 0) edgeHits++;
    }
    const edgeSupport = edgeHits / boundaryPixels.length;

    // Calculate shape metrics
    const metrics = this.calculateShapeMetrics(
      hull,
      pixels,
      centerX,
      centerY,
      perimeter,
      boundingArea,
      edgeSupport
    );
    const combinedMetrics = { ...metrics, ...radial, polygonSides };
    
    // Classify based on metrics
    const classification = this.classifyBasedOnMetrics(combinedMetrics, corners);

    if (!classification) return null;

    // Calculate confidence using edges and corner information
    const confidence = this.calculateConfidence(classification, combinedMetrics, area, corners);

    return {
      type: classification,
      confidence: Math.min(1.0, Math.max(0.5, confidence)),
      boundingBox: {
        x: minX,
        y: minY,
        width,
        height
      },
      center: {x: centerX, y: centerY},
      area
    };
  }

  /**
   * Harris corner detection for identifying vertices
   */
  private detectCorners(hull: Array<{x: number; y: number}>, centerX: number, centerY: number): Array<{x: number; y: number}> {
    if (hull.length < 3) return [];

    const candidates: Array<{ p: {x: number; y: number}; score: number }> = [];
    for (let i = 0; i < hull.length; i++) {
      const prev = hull[(i - 1 + hull.length) % hull.length];
      const cur = hull[i];
      const next = hull[(i + 1) % hull.length];

      const v1x = prev.x - cur.x;
      const v1y = prev.y - cur.y;
      const v2x = next.x - cur.x;
      const v2y = next.y - cur.y;
      const n1 = Math.sqrt(v1x * v1x + v1y * v1y);
      const n2 = Math.sqrt(v2x * v2x + v2y * v2y);
      if (n1 < 1 || n2 < 1) continue;

      const dot = (v1x * v2x + v1y * v2y) / (n1 * n2);
      const clamped = Math.max(-1, Math.min(1, dot));
      const angle = Math.acos(clamped);

      // Smaller internal angle => stronger corner.
      const curvature = Math.PI - angle;
      if (curvature > 0.65) {
        const radial = Math.sqrt((cur.x - centerX) ** 2 + (cur.y - centerY) ** 2);
        candidates.push({ p: cur, score: curvature * radial });
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    const corners: Array<{x: number; y: number}> = [];
    const minCornerDistance = 10;

    for (const c of candidates) {
      let tooClose = false;
      for (const kept of corners) {
        const dx = kept.x - c.p.x;
        const dy = kept.y - c.p.y;
        if (dx * dx + dy * dy < minCornerDistance * minCornerDistance) {
          tooClose = true;
          break;
        }
      }
      if (!tooClose) corners.push(c.p);
      if (corners.length >= 10) break;
    }

    return corners;
  }

  /**
   * Extract boundary pixels and estimate perimeter from 4-neighbor exposure.
   */
  private extractBoundaryAndPerimeter(pixels: Array<{x: number; y: number}>): {
    boundaryPixels: Array<{x: number; y: number}>;
    perimeter: number;
  } {
    const pixelSet = new Set(pixels.map((p) => `${p.x},${p.y}`));
    const boundaryPixels: Array<{x: number; y: number}> = [];
    let perimeter = 0;

    const dirs = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ];

    for (const p of pixels) {
      let exposed = 0;
      for (const [dx, dy] of dirs) {
        if (!pixelSet.has(`${p.x + dx},${p.y + dy}`)) {
          exposed++;
        }
      }
      if (exposed > 0) {
        boundaryPixels.push(p);
        perimeter += exposed;
      }
    }

    return { boundaryPixels, perimeter };
  }

  /**
   * Analyze boundary radius by angle bins for robust circle detection.
   */
  private analyzeRadialSignature(
    boundaryPixels: Array<{x: number; y: number}>,
    centerX: number,
    centerY: number
  ): { coverage: number; radialCV: number; radialDiff: number } {
    const binCount = 72;
    const bins: number[][] = Array.from({ length: binCount }, () => []);

    for (const p of boundaryPixels) {
      const dx = p.x - centerX;
      const dy = p.y - centerY;
      const r = Math.sqrt(dx * dx + dy * dy);
      let angle = Math.atan2(dy, dx);
      if (angle < 0) angle += 2 * Math.PI;
      const idx = Math.min(binCount - 1, Math.floor((angle / (2 * Math.PI)) * binCount));
      bins[idx].push(r);
    }

    const radialMeans: number[] = [];
    let filled = 0;
    for (const b of bins) {
      if (b.length === 0) {
        radialMeans.push(0);
        continue;
      }
      filled++;
      const sum = b.reduce((acc, v) => acc + v, 0);
      radialMeans.push(sum / b.length);
    }

    const coverage = filled / binCount;
    const validRadii = radialMeans.filter((v) => v > 0);
    if (validRadii.length < 8) {
      return { coverage, radialCV: 1, radialDiff: 1 };
    }

    const mean = validRadii.reduce((acc, v) => acc + v, 0) / validRadii.length;
    const variance = validRadii.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / validRadii.length;
    const std = Math.sqrt(variance);
    const radialCV = mean > 0 ? std / mean : 1;

    // Adjacent angular radius change; low for circles, high for stars/polygons.
    let diffSum = 0;
    let diffCount = 0;
    for (let i = 0; i < binCount; i++) {
      const a = radialMeans[i];
      const b = radialMeans[(i + 1) % binCount];
      if (a > 0 && b > 0) {
        diffSum += Math.abs(a - b);
        diffCount++;
      }
    }
    const radialDiff = diffCount > 0 && mean > 0 ? (diffSum / diffCount) / mean : 1;

    return { coverage, radialCV, radialDiff };
  }

  /**
   * Estimate polygon sides by simplifying hull contour.
   */
  private estimatePolygonSides(hull: Array<{x: number; y: number}>, epsilon: number): number {
    if (hull.length < 3) return hull.length;
    const closed = [...hull, hull[0]];
    const simplified = this.simplifyRDP(closed, epsilon);
    // Last point duplicates the first in closed polyline.
    return Math.max(3, simplified.length - 1);
  }

  /**
   * Ramer-Douglas-Peucker simplification for 2D points.
   */
  private simplifyRDP(points: Array<{x: number; y: number}>, epsilon: number): Array<{x: number; y: number}> {
    if (points.length < 3) return points;

    const start = points[0];
    const end = points[points.length - 1];
    let maxDist = 0;
    let index = 0;

    for (let i = 1; i < points.length - 1; i++) {
      const dist = this.pointLineDistance(points[i], start, end);
      if (dist > maxDist) {
        maxDist = dist;
        index = i;
      }
    }

    if (maxDist > epsilon) {
      const left = this.simplifyRDP(points.slice(0, index + 1), epsilon);
      const right = this.simplifyRDP(points.slice(index), epsilon);
      return [...left.slice(0, -1), ...right];
    }

    return [start, end];
  }

  private pointLineDistance(p: {x: number; y: number}, a: {x: number; y: number}, b: {x: number; y: number}): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const denom = Math.sqrt(dx * dx + dy * dy);
    if (denom === 0) {
      const px = p.x - a.x;
      const py = p.y - a.y;
      return Math.sqrt(px * px + py * py);
    }
    return Math.abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / denom;
  }

  /**
   * Graham scan algorithm for convex hull
   */
  private getConvexHull(pixels: Array<{x: number; y: number}>) {
    if (pixels.length < 3) return pixels;

    // Find starting point (lowest, leftmost)
    let start = pixels[0];
    for (const p of pixels) {
      if (p.y > start.y || (p.y === start.y && p.x < start.x)) {
        start = p;
      }
    }

    // Sort by polar angle
    const sorted = pixels.slice().sort((a, b) => {
      const angleA = Math.atan2(a.y - start.y, a.x - start.x);
      const angleB = Math.atan2(b.y - start.y, b.x - start.x);
      
      if (Math.abs(angleA - angleB) < 0.001) {
        // If same angle, sort by distance
        const distA = Math.pow(a.x - start.x, 2) + Math.pow(a.y - start.y, 2);
        const distB = Math.pow(b.x - start.x, 2) + Math.pow(b.y - start.y, 2);
        return distA - distB;
      }
      return angleA - angleB;
    });

    // Build hull with more lenient cross product check
    const hull: Array<{x: number; y: number}> = [];
    for (const p of sorted) {
      while (hull.length > 1) {
        const o = hull[hull.length - 2];
        const a = hull[hull.length - 1];
        const cross = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x);
        if (cross <= 0) hull.pop();
        else break;
      }
      hull.push(p);
    }

    return hull;
  }

  /**
   * Calculate shape metrics for classification
   */
  private calculateShapeMetrics(
    hull: Array<{x: number; y: number}>,
    pixels: Array<{x: number; y: number}>,
    centerX: number,
    centerY: number,
    perimeter: number,
    boundingArea: number,
    edgeSupport: number
  ) {
    const vertices = hull.length;
    
    // Calculate distances from center
    const distances: number[] = [];
    for (const p of hull) {
      const dx = p.x - centerX;
      const dy = p.y - centerY;
      distances.push(Math.sqrt(dx * dx + dy * dy));
    }

    // Calculate metrics
    const avgRadius = distances.length > 0 ? distances.reduce((a, b) => a + b, 0) / distances.length : 0;
    const radiusVariance = distances.length > 0 ? 
      Math.sqrt(distances.reduce((sum, d) => sum + Math.pow(d - avgRadius, 2), 0) / distances.length) : 0;
    const radiusCV = avgRadius > 0 ? radiusVariance / avgRadius : 999;

    const area = pixels.length;
    const circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;

    // Aspect ratio
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of hull) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    
    const hullWidth = (maxX - minX) || 1;
    const hullHeight = (maxY - minY) || 1;
    const aspectRatio = Math.max(hullWidth, hullHeight) / Math.min(hullWidth, hullHeight);

    let maxRadiusDiff = 0;
    if (vertices > 2 && distances.length > 0) {
      const minDist = Math.min(...distances);
      const maxDist = Math.max(...distances);
      maxRadiusDiff = maxDist - minDist;
    }

    // Solidity (ratio of area to convex hull area)
    let hullArea = 0;
    for (let i = 0; i < hull.length; i++) {
      const p1 = hull[i];
      const p2 = hull[(i + 1) % hull.length];
      hullArea += (p1.x * p2.y - p2.x * p1.y) / 2;
    }
    hullArea = Math.abs(hullArea);
    const solidity = hullArea > 0 ? area / hullArea : 0;
    const extent = area / Math.max(1, boundingArea);

    return {
      vertices,
      avgRadius,
      radiusCV,
      circularity,
      aspectRatio,
      area,
      perimeter,
      maxRadiusDiff,
      solidity,
      extent,
      edgeSupport
    };
  }

  /**
   * Classify shape based on metrics
   */
  private classifyBasedOnMetrics(metrics: any, corners: Array<{x: number; y: number}>): "circle" | "triangle" | "rectangle" | "pentagon" | "star" | null {
    const {radiusCV, circularity, solidity, extent, edgeSupport, maxRadiusDiff, avgRadius, coverage, radialCV, radialDiff, polygonSides} = metrics;
    const cornerCount = corners.length;
    const radialSwing = avgRadius > 0 ? maxRadiusDiff / avgRadius : 0;

    if (edgeSupport < 0.18) {
      return null;
    }

    // Pentagon: medium extent, strong convexity, and clear multi-corner profile.
    if (
      (polygonSides === 5 || (cornerCount >= 4 && cornerCount <= 6)) &&
      extent >= 0.48 &&
      extent <= 0.78 &&
      solidity > 0.84 &&
      radialDiff >= 0.1
    ) {
      return "pentagon";
    }

    // Circle via radial-angle consistency (strongest cue for this challenge).
    if (
      polygonSides >= 7 &&
      cornerCount <= 4 &&
      coverage > 0.75 &&
      radialCV < 0.14 &&
      radialDiff < 0.22 &&
      extent > 0.64 &&
      extent < 0.9
    ) {
      return "circle";
    }

    // Star: non-convex and spiky silhouette.
    if (solidity < 0.78 && extent < 0.52 && (cornerCount >= 6 || radialSwing > 0.55)) {
      return "star";
    }

    // Circle: compact, round and not spiky.
    if (
      polygonSides >= 7 &&
      cornerCount <= 4 &&
      circularity > 0.6 &&
      extent > 0.62 &&
      extent < 0.9 &&
      radialSwing < 0.5 &&
      radialCV < 0.2
    ) {
      return "circle";
    }

    // Rectangle: high fill ratio in bounding box.
    if (solidity > 0.92 && extent > 0.75) {
      return "rectangle";
    }

    // Triangle: lower extent than rectangles/circles, but still convex.
    if (extent >= 0.38 && extent < 0.66 && solidity > 0.9 && radialSwing > 0.45) {
      return "triangle";
    }

    // Pentagon: convex medium extent with moderate corner count.
    if (extent >= 0.5 && extent <= 0.76 && solidity > 0.84 && cornerCount >= 4 && cornerCount <= 7) {
      return "pentagon";
    }

    // Fallbacks by strongest remaining cues.
    if (polygonSides === 5) return "pentagon";
    if (polygonSides >= 7 && cornerCount <= 4 && coverage > 0.7 && radialCV < 0.18 && radialDiff < 0.3) return "circle";
    if (polygonSides >= 7 && cornerCount <= 4 && circularity > 0.65 && extent > 0.6 && radialSwing < 0.55) return "circle";
    if (extent > 0.78 && solidity > 0.88) return "rectangle";
    if (solidity < 0.8 && extent < 0.56) return "star";
    if (extent < 0.64) return "triangle";
    return "pentagon";
  }

  /**
   * Calculate confidence score
   */
  private calculateConfidence(type: string, metrics: any, area: number, corners: Array<{x: number; y: number}>): number {
    const {radiusCV, circularity, vertices, solidity, extent, edgeSupport, coverage, radialCV, radialDiff} = metrics;
    const cornerCount = corners.length;
    let confidence = 0.5;

    switch (type) {
      case "circle":
        confidence = 0.45
          + circularity * 0.22
          + (1 - Math.min(radiusCV, 1)) * 0.08
          + Math.max(0, 1 - Math.min(radialCV / 0.25, 1)) * 0.18
          + Math.max(0, 1 - Math.min(radialDiff / 0.35, 1)) * 0.12
          + Math.min(coverage, 1) * 0.08;
        break;
      case "triangle":
        confidence = 0.5 + (cornerCount === 3 ? 0.22 : 0.1) + solidity * 0.12 + extent * 0.08;
        break;
      case "rectangle":
        confidence = 0.58 + (cornerCount === 4 ? 0.15 : 0.07) + solidity * 0.14 + extent * 0.08;
        break;
      case "pentagon":
        confidence = 0.52 + (cornerCount === 5 ? 0.2 : 0.1) + solidity * 0.1 + extent * 0.05;
        break;
      case "star":
        confidence = 0.55 + (cornerCount >= 8 ? 0.2 : 0.1) + (1 - Math.min(solidity, 1)) * 0.1 + edgeSupport * 0.05;
        break;
    }

    // Adjust based on area
    if (area < 50) confidence -= 0.15;
    else if (area < 100) confidence -= 0.05;
    else if (area > 10000) confidence -= 0.05;

    return Math.min(0.99, Math.max(0.5, confidence));
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
