import AppKit
import PDFKit
import Foundation

guard CommandLine.arguments.count > 1, (CommandLine.arguments.count - 1) % 3 == 0 else {
    fatalError("Usage: render_selected.swift input.pdf one_based_page output.png [triplets...]")
}
for start in stride(from: 1, to: CommandLine.arguments.count, by: 3) {
    let input = URL(fileURLWithPath: CommandLine.arguments[start])
    let number = Int(CommandLine.arguments[start + 1])!
    let output = URL(fileURLWithPath: CommandLine.arguments[start + 2])
    guard let document = PDFDocument(url: input), let page = document.page(at: number - 1) else {
        fatalError("Cannot read requested PDF page")
    }
    let image = page.thumbnail(of: NSSize(width: 1200, height: 1700), for: .mediaBox)
    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Cannot render PNG")
    }
    try png.write(to: output)
    print(output.path)
}
