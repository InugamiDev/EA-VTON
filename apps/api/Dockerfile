FROM node:20-alpine

WORKDIR /app

COPY package.json .
RUN npm install --production

COPY src/ /app/src/

EXPOSE 3001

CMD ["node", "src/index.js"]
