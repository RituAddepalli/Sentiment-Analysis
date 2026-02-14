
// for local 

// import { Component } from '@angular/core';
// import { HttpClient } from '@angular/common/http';

// @Component({
//   selector: 'app-root',
//   templateUrl: './app.component.html',
//   styleUrls: ['./app.component.scss']
// })
// export class AppComponent {
//   userText: string = '';
//   result: any = null;
//   loading: boolean = false;

//   constructor(private http: HttpClient) {}

//   analyze() {
//     if (!this.userText.trim()) {
//       alert('Please enter some text!');
//       return;
//     }

//     this.loading = true;

//     this.http.post<any>('http://127.0.0.1:5000/analyze', { text: this.userText }).subscribe({
//       next: (res) => {
//         this.result = res;
//         this.loading = false;
//       },
//       error: (err) => {
//         alert('Error calling API: ' + err.message);
//         this.loading = false;
//       }
//     });
//   }

//   getResultColor(): string {
//     if (!this.result) return '#e6f0ff';
//     switch (this.result.sentiment.toLowerCase()) {
//       case 'positive': return '#d4edda';
//       case 'negative': return '#f8d7da';
//       case 'neutral': return '#fff3cd';
//       default: return '#e6f0ff';
//     }
//   }
// }














// import { Component } from '@angular/core';
// import { HttpClient } from '@angular/common/http';

// @Component({
//   selector: 'app-root',
//   templateUrl: './app.component.html',
//   styleUrls: ['./app.component.scss']
// })
// export class AppComponent {
//   userText: string = '';
//   result: any = null;
//   loading: boolean = false;

//   constructor(private http: HttpClient) {}

//   analyze() {
//     if (!this.userText.trim()) {
//       alert('Please enter some text!');
//       return;
//     }

//     this.loading = true;

//     this.http.post<any>('http://127.0.0.1:5000/analyze', { text: this.userText }).subscribe({
//       next: (res) => {
//         this.result = res;
//         this.loading = false;
//       },
//       error: (err) => {
//         alert('Error calling API: ' + err.message);
//         this.loading = false;
//       }
//     });
//   }

//   getResultColor(): string {
//     if (!this.result) return '#e6f0ff';
//     switch (this.result.sentiment.toLowerCase()) {
//       case 'positive': return '#d4edda'; // greenish
//       case 'negative': return '#f8d7da'; // reddish
//       case 'neutral': return '#fff3cd';  // yellowish
//       default: return '#e6f0ff';
//     }
//   }

//   // ✅ NEW: Helper to calculate polarity percentage for progress bars
//   getPolarityPercent(): number {
//     if (!this.result || this.result.polarity === undefined || this.result.polarity === null) {
//       return 0;
//     }
//     return Math.min(100, Math.abs(this.result.polarity) * 100);
//   }
// }







// import { Component } from '@angular/core';
// import { HttpClient } from '@angular/common/http';

// @Component({
//   selector: 'app-root',
//   templateUrl: './app.component.html',
//   styleUrls: ['./app.component.scss']
// })
// export class AppComponent {
//   userText: string = '';
//   result: any = null;
//   loading: boolean = false;

//   constructor(private http: HttpClient) {}

//   analyze() {
//     if (!this.userText.trim()) {
//       alert('Please enter some text!');
//       return;
//     }

//     this.loading = true;

//     this.http.post<any>('http://127.0.0.1:5000/analyze', { text: this.userText }).subscribe({
//       next: (res) => {
//         this.result = res;
//         this.loading = false;
//       },
//       error: (err) => {
//         alert('Error calling API: ' + err.message);
//         this.loading = false;
//       }
//     });
//   }

//   getResultColor(): string {
//     if (!this.result) return '#e6f0ff';
//     switch (this.result.sentiment.toLowerCase()) {
//       case 'positive': return '#d4edda';
//       case 'negative': return '#f8d7da';
//       case 'neutral': return '#fff3cd';
//       default: return '#e6f0ff';
//     }
//   }

//   // Polarity for progress bar (0-100)
//   getPolarityPercent(): number {
//     if (!this.result || this.result.polarity === undefined || this.result.polarity === null) {
//       return 0;
//     }
//     return Math.min(100, Math.abs(this.result.polarity) * 100);
//   }
// }












import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment'; // <-- import environment

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  userText: string = '';
  result: any = null;
  loading: boolean = false;

  constructor(private http: HttpClient) {}

  analyze() {
    if (!this.userText.trim()) {
      alert('Please enter some text!');
      return;
    }

    this.loading = true;

    // <-- use environment.apiUrl instead of hardcoded localhost
    this.http.post<any>(`${environment.apiUrl}/analyze`, { text: this.userText }).subscribe({
      next: (res) => {
        this.result = res;
        this.loading = false;
      },
      error: (err) => {
        alert('Error calling API: ' + err.message);
        this.loading = false;
      }
    });
  }

  getResultColor(): string {
    if (!this.result) return '#e6f0ff';
    switch (this.result.sentiment.toLowerCase()) {
      case 'positive': return '#d4edda';
      case 'negative': return '#f8d7da';
      case 'neutral': return '#fff3cd';
      default: return '#e6f0ff';
    }
  }
}



// import { Component } from '@angular/core';
// import { HttpClient } from '@angular/common/http';

// @Component({
//   selector: 'app-root',
//   templateUrl: './app.component.html',
//   styleUrls: ['./app.component.scss']
// })
// export class AppComponent {
//   userText: string = '';
//   result: { sentiment: string; polarity: number } | null = null;
//   loading: boolean = false; // <-- add this

//   constructor(private http: HttpClient) {}

//   analyze() {
//     if (!this.userText.trim()) {
//       alert('Please enter some text!');
//       return;
//     }

//     this.loading = true; // <-- start loading
//     this.result = null;

//     const payload = { text: this.userText };

//     this.http.post<any>('http://127.0.0.1:5000/analyze', payload).subscribe({
//       next: (res) => {
//         this.result = { sentiment: res.sentiment, polarity: res.polarity };
//         this.loading = false; // <-- end loading
//       },
//       error: (err) => {
//         alert('Error calling API: ' + err.message);
//         this.loading = false; // <-- end loading on error
//       }
//     });
//   }

//   // <-- add this method
//   getResultColor(): string {
//     if (!this.result) return '';
//     switch (this.result.sentiment.toLowerCase()) {
//       case 'positive': return '#d4edda';
//       case 'negative': return '#f8d7da';
//       case 'neutral': return '#fff3cd';
//       default: return '';
//     }
//   }
// }













// import { Component } from '@angular/core';
// import { HttpClient } from '@angular/common/http';

// @Component({
//   selector: 'app-root',
//   templateUrl: './app.component.html'
// })
// export class AppComponent {
//   userText: string = '';
//   result: { sentiment: string; polarity: number } | null = null;

//   constructor(private http: HttpClient) {}

//   analyze() {
//     if (!this.userText.trim()) {
//       alert('Please enter some text!');
//       return;
//     }

//     const payload = { text: this.userText };

//     this.http.post<any>('http://127.0.0.1:5000/analyze', payload).subscribe({
//       next: (res) => {
//         this.result = { sentiment: res.sentiment, polarity: res.polarity };
//       },
//       error: (err) => alert('Error calling API: ' + err.message)
//     });
//   }

//   getResultColor(): string {
//     if (!this.result) return '#e6f0ff'; // default light blue
//     switch (this.result.sentiment.toLowerCase()) {
//       case 'positive':
//         return '#d4edda'; // green
//       case 'negative':
//         return '#f8d7da'; // red
//       case 'neutral':
//         return '#fff3cd'; // yellow/orange
//       default:
//         return '#e6f0ff';
//     }
//   }
// }
