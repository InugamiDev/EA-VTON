# EA-VTON - Script thuyết trình cho bản present

File này đi cùng với:
- [present-weekly-report-2026-04-19.html](/Users/inugami/Documents/GitHub/research-try-out/reports/present-weekly-report-2026-04-19.html)

Mục tiêu của file này:
- giúp nói mạch lạc hơn khi trình bày;
- không cần đọc lại nguyên văn slide;
- tách rõ 2 nhánh của hệ thống: `size recommendation` và `virtual try-on`.

Thời lượng gợi ý:
- Bản ngắn: 5-6 phút.
- Bản đầy đủ: 7-8 phút.
- Nếu đi cả phụ lục: 9-10 phút.

Nguyên tắc khi nói:
- Mỗi slide chỉ có 1 ý chính.
- Không cần đọc hết bullet trên slide.
- Nếu thời gian ngắn, chỉ nói 8 slide chính.
- Phụ lục dùng khi có người hỏi thêm.

## Mở đầu 20-30 giây

Hôm nay em báo cáo tiến độ của EA-VTON tại mốc ngày 19 tháng 4 năm 2026.

Điểm quan trọng nhất ở mốc này là nhóm đã tách rõ hệ thống thành hai nhánh khác nhau, rồi chạy lại nhánh size để có số thật thay vì chỉ nói theo estimation.

Nhánh thứ nhất là `size recommendation`, đây là phần nghiên cứu chính.
Nhánh thứ hai là `virtual try-on`, đây là phần kỹ thuật để tạo ảnh thử đồ.

Em sẽ đi lần lượt theo 3 ý:
- vì sao nhóm chọn hướng này;
- nhóm đang làm hai nhánh đó như thế nào;
- và hiện tại mình đánh giá kết quả ra sao.

## Slide 1 - Mở bài

### Ý chính

Ở mốc báo cáo này, mục tiêu không chỉ là thêm thí nghiệm, mà là gom các phần đang có thành một hệ thống có thể giải thích và đo được.

### Script gợi ý

Ở slide đầu tiên, ý em muốn chốt là dự án đã đi từ nhiều thí nghiệm rời rạc sang một cấu trúc rõ ràng hơn, và bắt đầu thay các claim ước lượng bằng số đã chạy lại.

Trước đây mình có nhiều phần làm song song, nhưng chưa nói thật rõ phần nào là nghiên cứu chính, phần nào là phần kỹ thuật phục vụ sản phẩm.

Ở mốc báo cáo này, nhóm thống nhất lại câu chuyện của hệ thống: đầu vào là ảnh người dùng và thông tin cơ bản, sau đó hệ thống đi ra hai loại đầu ra khác nhau, một là gợi ý size, hai là ảnh thử đồ.

Điểm mới hơn so với bản deck cũ là nhánh size đã có một vòng train, eval và tuning thật trong repo, và research endpoint hiện cũng đã trỏ sang candidate tuned.
Còn nhánh VTON thì vẫn được giữ ở trạng thái pipeline chạy được chứ chưa thổi lên thành kết quả đẹp.

### Câu chuyển slide

Để tránh hiểu nhầm, slide tiếp theo em tách rất rõ hai nhánh này.

## Slide 2 - Hệ thống có 2 nhánh khác nhau

### Ý chính

Hai nhánh dùng chung đầu vào ở mức sản phẩm, nhưng là hai bài toán khác nhau.

### Script gợi ý

Ở đây em muốn làm rõ rằng `size recommendation` và `VTON` không nên bị gộp thành một thứ.

Nhánh `size recommendation` nhận các đặc trưng như chiều cao, cân nặng, đặc trưng cơ thể và loại sản phẩm, rồi dự đoán ra size, độ tin cậy và một số đo ước lượng.

Trong khi đó, nhánh `VTON` nhận ảnh người và ảnh quần áo để sinh ra ảnh mặc thử.

Vì vậy, cách đo của hai nhánh cũng khác nhau.
Nhánh size đo bằng accuracy, within-1, bias và ESS.
Nhánh VTON đo bằng thời gian chạy, chất lượng ảnh và độ ổn định vùng mặt, vùng da.

Nếu không tách hai nhánh này, phần đánh giá rất dễ bị lẫn giữa một bên là bài toán dự đoán và một bên là bài toán dựng ảnh.

### Câu chuyển slide

Sau khi tách ra như vậy, em sẽ nói trước về nhánh size vì đây là phần nghiên cứu chính.

## Slide 3 - Size recommendation là phần nghiên cứu chính

### Ý chính

Lý do chọn hướng này là vì nút thắt lớn nhất hiện tại là lệch phân bố giữa dữ liệu nguồn và nhóm người dùng mục tiêu.

### Script gợi ý

Với nhánh size, vấn đề chính không phải là thiếu mô hình phức tạp, mà là dữ liệu mình đang có và nhóm người dùng mình muốn phục vụ không hoàn toàn giống nhau.

Nếu học trực tiếp trên dữ liệu nguồn, mô hình có thể dự đoán ổn trên tập nguồn nhưng bị lệch khi áp vào nhóm mục tiêu.

Vì vậy nhóm chọn hướng `reweighting`, tức là không bỏ dữ liệu cũ đi, mà tăng trọng số cho những mẫu giống nhóm mục tiêu hơn.

Sau bước đó, nhóm huấn luyện và so sánh các mô hình như `GBM` và `MLP`.
Trong giai đoạn hiện tại, `GBM` vẫn là baseline mạnh và ổn định hơn.

Hiện tại mình có thể hiểu nhánh này theo hai lớp:
- `6` biến thể gốc để chứng minh câu chuyện nghiên cứu;
- và `2` tuned candidates để chọn model thực dụng hơn cho service.

Ở slide này em cũng có đưa công thức vào luôn để người nghe thấy rõ là mình đang dùng density ratio thật và weighted loss thật, chứ không chỉ nói bằng ý tưởng.

Điểm cần chốt ở slide này là:
đây là phần mang giá trị nghiên cứu chính, vì nó xử lý trực tiếp bài toán thích nghi phân bố giữa source và target.

### Câu chuyển slide

Còn nhánh VTON thì khác, đó là phần thiên về kỹ thuật triển khai sản phẩm hơn.

## Slide 4 - VTON là phần kỹ thuật phục vụ sản phẩm

### Ý chính

VTON không phải phần nghiên cứu cốt lõi ở mốc báo cáo này; nó là nhánh phục vụ trải nghiệm sản phẩm và được tổ chức để nối vào pipeline chung.

### Script gợi ý

Ở slide này em tách riêng hai việc.

Một bên là workflow của nhánh size trong thư mục `research`, gồm tiền xử lý dữ liệu, gán trọng số, huấn luyện mô hình và đánh giá.

Bên còn lại là nhánh `VTON` và phần serving, tức là nhận request từ web, điều phối qua API, rồi gọi các service như feature estimation, recommendation và VTON.

Điểm em muốn nhấn mạnh là:
`VTON` quan trọng về mặt sản phẩm vì nó tạo ra đầu ra trực quan cho người dùng, nhưng nó không phải nơi mình đang đặt trọng tâm đóng góp phương pháp.

Ở đây nhóm ưu tiên tận dụng lại các khối CV và pipeline có sẵn, rồi benchmark chúng một cách có hệ thống.

### Câu chuyển slide

Khi đã tách hai nhánh như vậy, phần đánh giá cũng phải tách riêng.

## Slide 5 - Khung đánh giá tách riêng cho 2 nhánh

### Ý chính

Không thể dùng một bộ metric chung cho cả size recommendation và VTON.

### Script gợi ý

Với nhánh size, tiêu chí quan trọng nhất là mô hình có giúp tăng chất lượng dự đoán trên nhóm mục tiêu hay không.

Ở đây nhóm đặt mục tiêu là:
- within-1 trên nhóm VN tăng khoảng 2 điểm phần trăm so với baseline;
- độ suy giảm trên full test không quá lớn;
- bias tiến gần về 0 hơn;
- và ESS của hướng `copula + PSIS` phải tốt hơn hướng trọng số độc lập.

Với nhánh VTON, mình không nói accuracy theo kiểu classification.
Thay vào đó, mình nhìn vào trade-off giữa thời gian chạy và chất lượng ảnh, đồng thời kiểm tra độ ổn định ở các vùng nhạy cảm như mặt và da.

Slide này quan trọng vì nó trả lời câu hỏi:
thế nào thì được xem là tiến bộ ở từng nhánh.

### Câu chuyển slide

Sau đó em đi vào kết quả hiện tại của nhánh size trước.

## Slide 6 - Kết quả thật hiện tại của size recommendation

### Ý chính

Kết quả hiện tại đã tốt hơn deck cũ, nhưng improvement vẫn là nhỏ và phải kể rất chặt chẽ.

### Script gợi ý

Đây là slide quan trọng nhất vì ở đây mình phải nói đúng sự thật của số liệu vừa chạy.

Trước khi đọc số, em sẽ giải thích rất ngắn tên các model trên chart để người nghe không bị rối.
`GBM baseline` là model không gán trọng số.
`GBM độc lập` là gán trọng số riêng cho chiều cao và cân nặng.
`GBM copula` là gán trọng số theo cặp chiều cao - cân nặng.
`Temper` nghĩa là làm mềm trọng số bằng lũy thừa nhỏ hơn 1 để giảm tác động của các mẫu quá cực đoan.

Nếu chỉ nhìn bộ số cũ thì câu chuyện còn khá xấu.
Nhưng sau khi em chạy thêm một vòng tuning ngay trong repo, có hai ứng viên mới đáng chú ý.

Ứng viên thứ nhất là `gbm_copula_tempered_a075`.
Nó nâng `VN-range Within-1` lên `70.97%`, trong khi `Full-test Within-1` vẫn giữ ở `83.18%`.
Đây là cấu hình em nghiêng về hơn nếu mình ưu tiên đúng metric mục tiêu và vẫn muốn giữ logic `copula + PSIS`.
Đây cũng là candidate hiện đã được nối vào research endpoint của recommendation service.

Ứng viên thứ hai là `GBM độc lập + temper (gbm_indep_tempered_a05)`.
Nó cho `VN-range Top-1` cao nhất là `51.46%`, và bias cũng đỡ âm hơn.
Nếu sau này mình muốn ưu tiên exact size hơn là within-1, đây cũng là một đối chứng rất đáng giữ.

Điểm em muốn sửa rõ so với deck cũ là phần `k-hat`.
Số `7.171` trước đó đến từ tập `full fit` chưa lọc theo nhánh train size.
Còn trên đúng split train đang dùng cho model size, `k-hat` của `copula + PSIS` gần bằng `0`.
Tức là phần chẩn đoán trước đó đã kể sai bối cảnh.

Chốt lại slide này:
- `GBM` vẫn là nhánh mạnh nhất;
- `tempered weighting` có tạo ra cải thiện thật;
- `MLP + CORN` hiện vẫn là `NO-GO`.

### Câu chuyển slide

Sau phần size, em chuyển qua VTON, và ở đây câu chuyện cũng phải nói trung thực như vậy.

## Slide 7 - Trạng thái thật hiện tại của VTON

### Ý chính

VTON hiện đã chạy được, nhưng chất lượng đầu ra vẫn chưa đủ để xem là một kết quả mạnh.

### Script gợi ý

Ở slide này em không nói theo kiểu “đã benchmark xong và chọn được cấu hình đẹp nhất”, vì hiện tại mình chưa có đủ cơ sở để nói như vậy.

Điều mình có thể nói chắc là:
backend `local_composite` đã được chạy lại thành công trên sample có sẵn.
Nó trả về latency khoảng `40ms` và confidence khoảng `0.45`.

Tức là ở mức kỹ thuật, pipeline có chạy được.
Nhưng ảnh nhìn vẫn còn kiểu dán chồng lên người, nên chưa đủ tự nhiên.

Ở phía còn lại, checkpoint trainable hiện tại cho ra output rất nhiễu, nên cũng chưa thể dùng làm demo chất lượng.

Vì vậy, cách kể đúng nhất ở slide này là:
nhánh VTON đã có `working pipeline`, nhưng chưa có `strong visual result`.

### Câu chuyển slide

Từ hai nhánh đó, em chốt lại hướng đi hiện tại ở slide cuối.

## Slide 8 - Kết luận

### Ý chính

Hiện tại mình đã có đủ số thật để khóa narrative đúng: size có improvement thật, VTON thì chưa.

### Script gợi ý

Phần chốt của em có ba ý.

Ý thứ nhất, ở nhánh `size recommendation`, mình đã có một improvement thật trong repo chứ không còn chỉ là kế hoạch.
Nếu cần chọn một candidate để kể trên deck, em sẽ chọn `gbm_copula_tempered_a075`.
Điểm mới là candidate này không chỉ nằm ở file eval, mà đã được set làm model mặc định cho research endpoint.

Ý thứ hai, `body regression` vẫn là một điểm sáng kỹ thuật khá rõ, vì Ridge đã đạt average MAE khoảng `2.10 cm`.

Ý thứ ba, ở nhánh `VTON`, mình vẫn phải nói rất thẳng là pipeline đã chạy được, nhưng chất lượng ảnh hiện tại chưa đủ để xem là benchmark đẹp.

Nếu nói ngắn gọn một câu để kết thúc thì là:
ở mốc báo cáo này, nhóm không đi tìm một ảnh đẹp bằng cách mượn model ngoài, mà dùng chính pipeline của repo để tìm ra một improvement nhỏ nhưng bảo vệ được, đồng thời giữ ranh giới rất rõ giữa phần đã chứng minh được và phần chưa.

Việc tiếp theo là chạy lại luồng API end-to-end với candidate mới này để khóa nốt phần pipeline-level reporting.

## Phụ lục 1 - Từ điển thuật ngữ dùng trong deck

Phần này dùng khi có người hỏi sâu.
Không cần trình bày hết nếu thời gian ngắn.

### Nhóm thuật ngữ về bài toán

- `Size recommendation`: phần dự đoán người dùng nên mặc size nào.
- `Virtual try-on` hoặc `VTON`: phần tạo ảnh mặc thử từ ảnh người và ảnh quần áo.
- `Source`: dữ liệu nguồn, tức dữ liệu mình đang có để học.
- `Target`: nhóm người dùng mục tiêu mà mình muốn mô hình phục vụ tốt hơn.
- `Prior`: thống kê ban đầu về nhóm mục tiêu, ví dụ chiều cao, cân nặng, hoặc phân bố cụm cơ thể.
- `Distribution shift`: dữ liệu nguồn và dữ liệu mục tiêu khác nhau, nên nếu học thẳng từ source thì dễ dự đoán sai khi áp vào target.
- `Body clusters`: các nhóm hình thể chính trong dân số mục tiêu, dùng để mô tả cấu trúc cơ thể ở mức tổng quát.

### Nhóm thuật ngữ về đầu vào và đặc trưng

- `Feature`: các thông tin đầu vào cho model, ví dụ chiều cao, cân nặng, BMI, tuổi, body type và category.
- `Body type`: kiểu dáng cơ thể ở mức nhãn, ví dụ hourglass hay pear, dùng như một feature phụ.
- `Category`: loại sản phẩm, ví dụ dress hay tops, vì cùng một cơ thể nhưng mỗi loại đồ có thể fit khác nhau.
- `BMI`: chỉ số khối cơ thể, tính từ chiều cao và cân nặng, dùng như một đặc trưng bổ sung.
- `Measurements`: các số đo cơ thể ước lượng như ngực, eo, hông.

### Nhóm thuật ngữ về phương pháp gán trọng số

- `Weighting`: gán mức quan trọng khác nhau cho từng mẫu train, thay vì coi mọi mẫu ngang nhau.
- `Density ratio`: tỉ số mật độ giữa target và source. Mẫu nào giống target hơn thì có trọng số cao hơn.
- `Copula density ratio`: cách tính density ratio nhưng nhìn chiều cao và cân nặng như một cặp có liên hệ, không tách rời từng biến.
- `PSIS`: Pareto Smoothed Importance Sampling, tức bước làm mềm phần đuôi của phân phối trọng số để các mẫu quá hiếm không chi phối quá mạnh quá trình học.
- `Temper`: làm mềm thêm trọng số bằng cách lấy lũy thừa nhỏ hơn 1. Nói dễ hiểu là giữ hướng ưu tiên, nhưng giảm độ gắt của weighting.
- `Alpha`: hệ số dùng trong temper. Alpha càng nhỏ thì trọng số càng được làm mềm mạnh hơn.
- `Max weight`: trọng số lớn nhất trong toàn bộ tập train. Nếu quá lớn thì dễ gây mất ổn định.
- `k-hat`: chỉ số chẩn đoán tail của PSIS. Nếu gần 0 thì ổn hơn; nếu quá lớn thì phần tail đang có vấn đề.
- `ESS`: Effective Sample Size, tức số mẫu hiệu dụng sau khi gán trọng số. ESS càng cao thì việc huấn luyện thường càng lành hơn.

### Nhóm thuật ngữ về model

- `GBM`: Gradient Boosting Machine. Đây là model học bằng nhiều cây quyết định nhỏ, mỗi vòng sửa lỗi của vòng trước. Nó thường rất mạnh với dữ liệu bảng.
- `GBM baseline`: cấu hình GBM không gán trọng số, dùng làm mốc so sánh chính.
- `GBM độc lập`: cấu hình GBM dùng trọng số độc lập cho chiều cao và cân nặng.
- `GBM copula`: cấu hình GBM dùng copula để tính trọng số theo cặp chiều cao - cân nặng.
- `GBM độc lập + temper`: bản độc lập nhưng đã làm mềm trọng số.
- `GBM copula + temper`: bản copula nhưng đã làm mềm trọng số, hiện là candidate chính trong deck.
- `MLP`: mạng neural nhiều lớp fully connected. Trong bài toán này, nó là đối chứng neural so với GBM.
- `Ordinal`: cách nhìn bài toán size như một chuỗi có thứ tự, ví dụ XS nhỏ hơn S và S nhỏ hơn M.
- `Ordinal loss`: hàm mất mát có xét đến thứ tự giữa các size, không coi các nhãn là rời rạc hoàn toàn.
- `CORN`: một dạng ordinal classification cho neural network. Trong repo này nó chưa vượt được các baseline GBM.
- `Baseline`: mốc so sánh gốc. Nếu model mới không hơn baseline thì chưa nên claim là cải thiện.
- `Candidate`: cấu hình đang được cân nhắc để dùng thật trong service.
- `Tuned candidate`: candidate đã qua một vòng tuning để tối ưu hơn cấu hình gốc.
- `NO-GO`: kết luận rằng một hướng chưa đủ tốt để chọn làm hướng chính ở thời điểm hiện tại.

### Nhóm thuật ngữ về metric

- `Top-1`: dự đoán đúng exact size.
- `Exact size`: đúng đúng một size, ví dụ người dùng cần size M thì model cũng phải ra M.
- `Within-1`: dự đoán đúng hoặc chỉ lệch tối đa 1 size, ví dụ cần M mà ra S hoặc L thì vẫn chấp nhận được.
- `Bias`: độ lệch có hướng của model. Bias âm nghĩa là model hay dự đoán nhỏ hơn thực tế; bias dương nghĩa là hay dự đoán lớn hơn thực tế.
- `Full test`: toàn bộ tập test, dùng để xem model có giữ được chất lượng chung hay không.
- `VN-range`: nhóm test gần với người dùng mục tiêu Việt Nam hơn, là nơi mình quan tâm nhất trong deck này.
- `Degradation`: mức giảm hiệu năng khi đổi từ baseline sang model mới.
- `Accuracy`: tỉ lệ dự đoán đúng exact size.
- `Confidence`: mức tự tin của model với dự đoán hiện tại.

### Nhóm thuật ngữ về ước lượng số đo

- `Body regression`: nhánh ước lượng số đo cơ thể từ các feature đơn giản.
- `Ridge`: một dạng linear regression có regularization, đang là model tốt nhất cho nhánh body regression trong repo này.
- `MAE`: Mean Absolute Error, tức sai số tuyệt đối trung bình. Ví dụ MAE 2 cm nghĩa là trung bình lệch khoảng 2 cm.
- `R²`: mức độ giải thích được biến thiên của dữ liệu. R² càng cao thì model giải thích càng tốt.

### Nhóm thuật ngữ về hệ thống

- `Pipeline`: toàn bộ chuỗi xử lý từ đầu vào đến đầu ra.
- `Working pipeline`: pipeline đã chạy được từ đầu đến cuối, nhưng chưa có nghĩa là chất lượng đã tốt.
- `Service`: một thành phần độc lập trong hệ thống, ví dụ recommendation service hay VTON service.
- `Recommendation service`: service chịu trách nhiệm trả kết quả gợi ý size.
- `Research endpoint`: endpoint dùng model nghiên cứu đã train, khác với các rule-based fallback đơn giản.
- `Serving`: phần đưa model vào chạy như một dịch vụ có thể gọi từ hệ thống.
- `End-to-end`: chạy trọn luồng từ request đầu vào đến kết quả cuối cùng.
- `Latency`: thời gian xử lý của hệ thống hay của một backend.

### Nhóm thuật ngữ riêng cho VTON

- `local_composite`: backend ghép ảnh đơn giản đang có trong repo. Nó chạy nhanh nhưng chất lượng ảnh còn hạn chế.
- `Checkpoint trainable`: model checkpoint đã train hoặc đang train để sinh ảnh thử đồ. Trong deck hiện tại, output của nó vẫn chưa đủ đẹp.
- `Visual readiness`: mức độ sẵn sàng để đem ảnh lên slide hay demo cho người khác xem.
- `Fallback`: phương án thay thế khi nhánh chính chưa đủ tốt hoặc chưa sẵn sàng.

### Vì sao chọn hướng này thay vì hướng khác

Nói ngắn gọn:
- tốt hơn `rule-based` vì đo được bằng số liệu và dễ cải thiện theo vòng;
- tốt hơn `trọng số độc lập` vì nhìn được liên hệ giữa chiều cao và cân nặng;
- tốt hơn `trọng số thô` vì có PSIS để giảm mất ổn định;
- thực tế hơn `neural-only` ở giai đoạn hiện tại vì GBM đang ổn định hơn;
- nhanh hơn việc chờ thêm dữ liệu mới, dù về dài hạn vẫn nên bổ sung dữ liệu mục tiêu.

## Phụ lục 2 - Các nghiên cứu gần tinh thần Lighter-X

Phần này dùng khi có người hỏi:
“Vì sao lại ưu tiên hướng nhẹ, mô-đun, và benchmark rõ như vậy?”

### Cách nói ngắn

Có một nhóm nghiên cứu gần tinh thần đó như `LightGCN`, `UltraGCN`, `JGCF`, `LightGCL`, `SVD-GCN`, `Less is More` và `Lighter-X`.

Điểm chung của các hướng này là:
- không mặc định mô hình càng nặng càng tốt;
- cố gắng tách bài toán thành các phần rõ hơn;
- chỉ giữ những tín hiệu thực sự quan trọng;
- và benchmark rất chặt.

### Cách nối về EA-VTON

Điều nhóm học từ các nghiên cứu này không phải là bê nguyên mô hình graph về dùng.

Thứ nhóm học là tư duy thiết kế:
- xác định đúng nút thắt chính;
- giữ hệ thống nhẹ và thay thế được;
- và tách rõ phần nào là research, phần nào là serving.

Với EA-VTON hiện tại, nút thắt chính là `distribution shift` ở nhánh size, chứ không phải graph quá nặng.

Vì vậy, hướng `copula + PSIS + GBM` vẫn sát bài toán của mình hơn.

## Chốt 15 giây

Nếu cần kết thúc rất nhanh, có thể nói:

EA-VTON hiện nên được nhìn như một hệ thống hai nhánh.
Nhánh size là phần nghiên cứu chính và hiện tại hướng `copula + PSIS + GBM` là phù hợp nhất để đi tiếp.
Nhánh VTON là phần kỹ thuật phục vụ sản phẩm, và mục tiêu trước mắt là benchmark ổn định giữa tốc độ và chất lượng.
