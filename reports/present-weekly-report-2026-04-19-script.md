# EA-VTON - Script thuyết trình cho bản present

File này đi cùng với:
- [present-weekly-report-2026-04-19.html](/Users/inugami/Documents/GitHub/research-try-out/reports/present-weekly-report-2026-04-19.html)

Mục tiêu của file này:
- giúp nói mạch lạc hơn khi trình bày;
- không đọc lại nguyên văn slide;
- tách rất rõ 2 nhánh: `size recommendation` và `virtual try-on`.

Thời lượng gợi ý:
- Bản ngắn: 5-6 phút.
- Bản đầy đủ: 7-8 phút.

## Mở đầu 20-30 giây

Hôm nay em báo cáo tiến độ của EA-VTON tại mốc ngày 19 tháng 4 năm 2026, với code snapshot chốt ở ngày 22 tháng 4.

Điểm chính của deck này là nhóm đã tách rõ hệ thống thành 2 nhánh khác nhau.
Nhánh thứ nhất là `size recommendation`, đây là phần nghiên cứu chính.
Nhánh thứ hai là `virtual try-on`, đây là phần kỹ thuật xử lý ảnh để tạo trải nghiệm trực quan.

Mục tiêu của giai đoạn này không phải là thêm thật nhiều thử nghiệm, mà là hợp nhất các phần đang có thành một pipeline hoàn chỉnh, có thể giải thích và có thể đo.

## Slide 1 - Title / Overview

### Ý chính

Deck này chốt lại rằng dự án có 2 nhánh riêng và chỉ một nhánh đã có improvement thật rõ.

### Script gợi ý

Ở slide đầu tiên, em muốn chốt ba ý.

Ý thứ nhất, EA-VTON không phải là một bài toán duy nhất, mà là một hệ thống có 2 nhánh: size recommendation và virtual try-on.

Ý thứ hai, ở mốc này nhóm đã có một kết quả thật ở nhánh size, cụ thể là VN Within-1 đạt 70.97 phần trăm sau vòng tuning GBM ngay trong repo.

Ý thứ ba, ở nhánh VTON thì mình chỉ nên nói trung thực là đã có pipeline chạy được, chứ chưa đủ để xem là một kết quả triển khai thực tế.

### Câu chuyển slide

Để tránh hiểu lầm, em tách luôn 2 nhánh này ra ngay ở slide tiếp theo.

## Slide 2 - System Overview

### Ý chính

Hai nhánh liên quan ở mức sản phẩm nhưng khác nhau về bản chất.

### Script gợi ý

Ở đây em muốn làm rõ là không nên gộp `size recommendation` với `VTON` thành một câu chuyện chung.

Nhánh size là bài toán dự đoán: đầu vào là các đặc trưng cơ thể và loại sản phẩm, đầu ra là size, confidence và một số measurements ước lượng.

Nhánh VTON là bài toán dựng ảnh: đầu vào là ảnh người và ảnh quần áo, đầu ra là ảnh mặc thử.

Vì vậy cách đánh giá cũng khác.
Nhánh size nhìn bằng Within-1, Top-1, Bias, ESS.
Nhánh VTON nhìn bằng latency, quality, usability.

### Câu chuyển slide

Sau khi tách hai nhánh ra, em đi vào nhánh size trước vì đây là phần nghiên cứu chính.

## Slide 3 - Size Recommendation

### Ý chính

Nút thắt của nhánh size là lệch phân bố giữa dữ liệu nguồn và nhóm người dùng mục tiêu.

### Script gợi ý

Với nhánh size, vấn đề chính không phải là thiếu mô hình thật phức tạp, mà là dữ liệu nguồn và nhóm người dùng mình muốn phục vụ không hoàn toàn giống nhau.

Nếu train trực tiếp trên dữ liệu nguồn, model dễ học ra quy luật có lợi cho source nhưng lệch trên target users.

Vì vậy nhóm chọn hướng re-weight dữ liệu theo target users.
Ở đây copula được dùng để modeling chiều cao và cân nặng như một cặp có liên hệ với nhau, thay vì xem hai biến này là độc lập.
Sau đó PSIS được dùng để làm mượt tail của trọng số, để vài điểm quá hiếm không kéo lệch toàn bộ quá trình học.

Phần so sánh model hiện tại tập trung vào hai family là GBM và MLP.
Ở giai đoạn này, GBM vẫn là family mạnh và ổn định hơn.

### Câu chuyển slide

Tiếp theo là chỗ rất quan trọng: research pipeline và serving pipeline khác nhau.

## Slide 4 - Pipeline vs Serving

### Ý chính

Research pipeline và serving pipeline có vai trò khác nhau, dù cuối cùng cùng gặp nhau ở API output.

### Script gợi ý

Ở slide này em tách rất rõ hai lớp.

Lớp thứ nhất là `research pipeline` của nhánh size: preprocess, weighting, train và evaluate.

Lớp thứ hai là `serving pipeline`: web nhận input, gateway điều phối, feature service trích xuất measurement, rồi size service và VTON service trả output cuối.

Điểm quan trọng là nhánh size đi theo hướng modeling và domain adaptation, còn nhánh VTON đi theo hướng CV integration.

### Câu chuyển slide

Sau đó, em đưa luôn flow end-to-end để thấy hai nhánh gặp nhau ở đâu.

## Slide 5 - System Flow

### Ý chính

Ở mức sản phẩm, người dùng chỉ thấy một flow end-to-end duy nhất.

### Script gợi ý

Luồng tổng thể là: người dùng upload input ở web, gateway nhận request, feature service trích xuất measurement và feature cần thiết, sau đó nhánh size dự đoán size còn nhánh VTON tạo ảnh thử đồ.

Cuối cùng hai đầu ra này được trả ngược lại cho người dùng như một kết quả thống nhất.

Đây là chỗ quan trọng để phân biệt giữa trải nghiệm người dùng và cấu trúc kỹ thuật phía sau.
Người dùng thấy một hệ thống, nhưng bên dưới là hai nhánh có logic khác nhau.

### Câu chuyển slide

Khi hai nhánh đã được tách như vậy, phần đánh giá cũng phải tách ra tương ứng.

## Slide 6 - Evaluation Framework (Size)

### Ý chính

Nhánh size được đo bằng metric định lượng và ưu tiên đúng trên target users.

### Script gợi ý

Với nhánh size, metric chính là Within-1 trên nhóm VN.
Tức là model đoán đúng hoặc chỉ lệch tối đa một size trên nhóm người dùng gần với mục tiêu.

Ngoài ra còn ba điểm cần giữ đồng thời.
Một là full test không được giảm quá nhiều.
Hai là bias phải tiến gần về 0.
Ba là ESS của cách weighting tốt phải cao hơn hướng đơn giản hơn.

Nói ngắn gọn, slide này trả lời câu hỏi: thế nào thì được xem là tiến bộ ở nhánh size.

### Câu chuyển slide

Tương tự như vậy, nhánh VTON cũng cần một khung đánh giá riêng.

## Slide 7 - Evaluation Framework (VTON)

### Ý chính

Nhánh VTON không đo bằng classification metric, mà đo bằng chất lượng ảnh và khả năng sử dụng.

### Script gợi ý

Với VTON, thứ mình quan tâm không phải accuracy mà là ảnh có đủ tốt để người dùng tin hay không.

Vì vậy các tiêu chí chính là:
- trade-off giữa latency và quality;
- độ ổn định ở vùng mặt và da;
- benchmark nhất quán giữa các backend;
- và usability thực tế của output.

Slide này quan trọng vì nó giúp mình không vô tình dùng tiêu chí của nhánh size để đánh giá một bài toán dựng ảnh.

### Câu chuyển slide

Sau hai slide framework, em đi vào kết quả thật hiện có của nhánh size.

## Slide 8 - Size Status (Performance)

### Ý chính

Nhánh size đã có improvement thật, nhưng improvement này là nhỏ và cần kể rất chặt chẽ.

### Script gợi ý

Nếu nhìn toàn bộ full test, GBM baseline vẫn đang giữ Within-1 tốt nhất là 83.37 phần trăm.
Điều đó có nghĩa là baseline vẫn rất mạnh và là mốc để kiểm soát degradation.

Nhưng nếu nhìn đúng vào nhóm mục tiêu, thì cấu hình `GBM copula tempered, alpha 0.75` hiện cho VN Within-1 tốt nhất là 70.97 phần trăm.

Nếu nhìn exact size, tức Top-1, thì cấu hình `independent + temper` hiện lại tốt nhất với 51.46 phần trăm.

Điểm em muốn chốt ở đây là:
GBM vẫn là nhánh mạnh nhất;
tuning có tạo ra improvement thật;
và neural models hiện chưa vượt được baseline.

### Câu chuyển slide

Sau performance, em tách riêng phần analysis để giải thích vì sao hướng này vẫn đáng giữ.

## Slide 9 - Size Status (Analysis)

### Ý chính

Phần analysis cho thấy hướng weighting hiện tại không chỉ tăng metric đích mà còn ổn định hơn về mặt huấn luyện.

### Script gợi ý

Ở đây có hai tín hiệu chính.

Tín hiệu thứ nhất là ESS tăng từ 10.95 phần trăm ở hướng independent lên 24.09 phần trăm ở copula cộng PSIS.
Điều này cho thấy tập train sau weighting usable hơn và ổn định hơn.

Tín hiệu thứ hai là body regression hiện khá ổn, với MAE khoảng 2.10 cm và R bình phương khoảng 0.837.
Nó chưa phải mô hình đo cơ thể hoàn hảo, nhưng đủ tốt để hỗ trợ nhánh size.

Ngoài ra, k-hat trên split train hiện đang dùng gần 0, nên phần chẩn đoán PSIS ở narrative hiện tại là lành mạnh.

### Câu chuyển slide

Sau khi chốt size, em chuyển sang nhánh VTON và sẽ giữ đúng cùng mức độ trung thực.

## Slide 10 - VTON Status

### Ý chính

VTON hiện là một working pipeline, chưa phải một kết quả visual mạnh.

### Script gợi ý

Ở nhánh VTON, local composite chạy rất nhanh, khoảng 40 mili giây, nhưng ảnh vẫn có cảm giác dán chồng.

Ở phía trainable model, output hiện còn nhiễu và chưa đủ tốt để dùng như benchmark chất lượng hay demo usable.

Vì vậy cách kể đúng nhất ở mốc này là:
mình đã có pipeline chạy được,
nhưng chưa có visual quality đủ mạnh để claim như một thành quả hoàn chỉnh.

### Câu chuyển slide

Từ hai nhánh đó, em chốt lại model đang giữ và việc cần làm tiếp.

## Slide 11 - Conclusion & Next Steps

### Ý chính

Nhánh size đã có candidate rõ ràng; nhánh VTON thì chưa.

### Script gợi ý

Kết luận hiện tại của em có ba ý.

Ý thứ nhất, candidate chính ở nhánh size là `GBM copula tempered, alpha 0.75`, vì nó cho VN Within-1 tốt nhất là 70.97 phần trăm.

Ý thứ hai, `independent + temper` vẫn nên được giữ làm baseline phụ nếu sau này mình muốn ưu tiên Top-1 hơn.

Ý thứ ba, việc cần làm tiếp là chạy end-to-end pipeline thật ở mức request thật và tiếp tục nâng chất lượng VTON, thay vì chỉ tinh chỉnh thêm slide.

### Câu chuyển slide

Cuối cùng là một slide ngắn về related research để đặt hướng làm hiện tại vào đúng bối cảnh thiết kế.

## Slide 12 - Related Research

### Ý chính

Nhóm có tham khảo các hướng nghiên cứu lightweight và decoupled, nhưng không triển khai trực tiếp các mô hình đó.

### Script gợi ý

Ở slide cuối này, em không đi sâu vào từng paper, mà chỉ chốt là nhóm có tham khảo các hướng như LightGCN, UltraGCN, JGCF, LightGCL, SVD-GCN và Lighter-X ở mức tư duy thiết kế.

Điểm chung của các hướng này là:
ưu tiên mô hình gọn hơn,
tách mô-đun rõ hơn,
và chỉ giữ phần tín hiệu thật sự cần thiết.

Điều mình học từ đó là cách thiết kế pipeline nhẹ, rõ vai trò từng khối, và dễ benchmark.
Nhưng EA-VTON hiện vẫn bám vào bài toán riêng là size adaptation cộng với virtual try-on serving.

## Từ điển thuật ngữ ngắn

- `GBM`: mô hình học bằng nhiều cây quyết định nối tiếp nhau; rất hợp với dữ liệu bảng.
- `MLP`: mạng fully-connected nhiều tầng; hiện chưa thắng GBM ở repo này.
- `Copula`: cách mô hình chiều cao và cân nặng cùng nhau thay vì tách riêng.
- `PSIS`: bước làm mượt tail của trọng số để tránh vài điểm cực đoan kéo lệch model.
- `ESS`: số mẫu hiệu dụng sau weighting; cao hơn thường nghĩa là ổn định hơn.
- `Within-1`: dự đoán đúng hoặc chỉ lệch tối đa 1 size.
- `Top-1`: dự đoán đúng exact size.
- `Bias`: xu hướng model đoán lệch có hệ thống về một phía.
- `VTON`: virtual try-on, tức nhánh tạo ảnh thử đồ.

## Một câu chốt cuối

Ở mốc này, điều quan trọng nhất không phải là có một ảnh demo đẹp, mà là mình đã có một improvement thật ở nhánh size bằng chính logic của repo, đồng thời biết rất rõ phần nào đã chứng minh được và phần nào thì chưa.
