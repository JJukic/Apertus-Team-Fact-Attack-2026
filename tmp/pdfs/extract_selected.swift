import Foundation
import PDFKit

let arguments = CommandLine.arguments
guard arguments.count >= 3,
      let document = PDFDocument(url: URL(fileURLWithPath: arguments[1])) else {
    fatalError("Usage: extract_selected.swift PDF page [page ...]")
}
var output: [[String: Any]] = []
for argument in arguments.dropFirst(2) {
    guard let number = Int(argument), let page = document.page(at: number - 1) else {
        fatalError("Invalid page")
    }
    output.append(["page_number": number, "text": page.string ?? ""])
}
let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
