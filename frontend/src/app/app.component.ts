
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
// import { environment } from '../environments/environment'; // <-- import environment

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

//     // <-- use environment.apiUrl instead of hardcoded localhost
//     this.http.post<any>(`${environment.apiUrl}/analyze`, { text: this.userText }).subscribe({
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

  setExample(text: string) {
    this.userText = text;
    this.result = null; // clear previous result
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
